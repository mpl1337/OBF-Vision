import asyncio
import re
from pathlib import Path
from typing import Any

import numpy as np

from app.config import save_config
from app.engine.arcface import ArcFaceReID
from app.engine.core import GLOBAL_CORE, Core, OVModel
from app.engine.emb_quality import (
    dedupe_greedy,
    filter_outliers_to_centroid,
    pick_prototypes,
)
from app.engine.face import FaceDB
from app.engine.yolo_face import YoloFaceAdapter
from app.state import ENROLL_LOCK, log_enroll
from app.utils import calc_blur_score, decode_image_bytes, hard_exit, utc_rfc1123

_ENROLL_PENDING = False


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def _model_tag(model_path: str) -> str:
    stem = Path(model_path).stem or "model"
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("_")
    return tag or "model"


def _default_db_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    rec_model_path = str(cfg.get("rec_model", "./models/w600k_mbf.xml"))
    tag = _model_tag(rec_model_path)

    out_db = cfg.get("enroll_out_db")
    out_clean = cfg.get("enroll_prune_out_db")

    if not out_db:
        out_db = f"./faces_db_{tag}.faiss"
    if not out_clean:
        out_clean = f"./faces_db_{tag}_clean.faiss"

    return {"out_db": Path(str(out_db)), "out_clean": Path(str(out_clean)), "tag": tag}


def enroll_build_db(cfg: dict[str, Any]) -> dict[str, Any]:
    log_enroll(">>> START: Building Face DB (FAISS+JSON).")

    device_default = str(cfg.get("device", "CPU"))
    device_det = str(cfg.get("device_face_det", device_default))
    device_reid = str(cfg.get("device_reid", device_default))
    rec_model_path = str(cfg.get("rec_model", "./models/w600k_mbf.xml"))
    rec_preprocess = str(cfg.get("rec_preprocess", "auto"))
    cfg_det = str(cfg.get("face_det_model", "") or "").strip()
    det_model_path = cfg_det if cfg_det else None
    det_candidates = ["./models/yolov8n-face.xml", "yolov8n-face.xml"]
    if not det_model_path or not Path(det_model_path).exists():
        det_model_path = None
        for c in det_candidates:
            if Path(c).exists():
                det_model_path = c
                break

    root = Path(str(cfg.get("enroll_root", "./enroll")))
    if not root.exists():
        msg = f"FEHLER: Enroll-Ordner '{root}' existiert nicht!"
        log_enroll(msg)
        raise RuntimeError(msg)

    if not Path(rec_model_path).exists():
        msg = f"FEHLER: ReID Modell fehlt: {rec_model_path}"
        log_enroll(msg)
        raise RuntimeError(msg)

    if not det_model_path:
        msg = "FEHLER: Face Detector (yolov8n-face.xml) fehlt!"
        log_enroll(msg)
        raise RuntimeError(msg)

    core = GLOBAL_CORE if GLOBAL_CORE else Core()

    det_conf = float(cfg.get("face_min_conf") or cfg.get("face_conf") or 0.30)
    log_enroll(f"   Detector: {det_model_path}  (device={device_det}, conf={det_conf})")
    det_adapter = YoloFaceAdapter(
        OVModel(core, det_model_path, device_det), conf_thres=det_conf
    )
    log_enroll("-> Detector geladen.")

    log_enroll(f"   ReID: {Path(rec_model_path).name}  (device={device_reid})")
    reid = ArcFaceReID(
        OVModel(core, rec_model_path, device_reid),
        preprocess=rec_preprocess,
    )
    log_enroll("-> ReID Modell geladen. Starte Verarbeitung.")
    log_enroll(f"   -> Preprocess resolved to: {reid.preprocess}")

    paths = _default_db_paths(cfg)
    target_db_path = Path(str(cfg.get("enroll_out_db") or paths["out_db"]))
    if target_db_path.suffix.lower() == ".json":
        target_db_path = target_db_path.with_suffix(".faiss")
    elif target_db_path.suffix == "":
        target_db_path = target_db_path.with_suffix(".faiss")

    case_insensitive = bool(cfg.get("enroll_case_insensitive_folders", True))
    groups: dict[str, list[Path]] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        key = p.name.lower() if case_insensitive else p.name
        groups.setdefault(key, []).append(p)

    people_groups = []
    for key, dir_list in groups.items():
        db_name = dir_list[0].name if case_insensitive else key
        people_groups.append((db_name, dir_list))

    total_stats = {"seen": 0, "added": 0, "errors": 0, "blur_skipped": 0}

    keep_q = float(cfg.get("enroll_keep_quantile", 0.25))
    dedup_th = float(
        cfg.get("enroll_dedup_sim") or cfg.get("enroll_dedupe_sim") or 0.995
    )
    proto_k = int(cfg.get("enroll_prototypes_k", 5))
    dedup_enable = bool(cfg.get("enroll_dedup_enable", True))
    blur_skip_enable = bool(cfg.get("enroll_blur_skip_enable", True))
    blur_thres = float(cfg.get("enroll_blur_var_thres", 60.0))

    temp_db = FaceDB(str(target_db_path))
    temp_db._create_empty()

    for name, dir_list in people_groups:
        log_enroll(f"Prüfe Person: {name}.")
        person_seen = 0
        person_embs: list[np.ndarray] = []

        for d in dir_list:
            for imgp in d.rglob("*"):
                if not _is_image(imgp):
                    continue

                total_stats["seen"] += 1
                person_seen += 1
                try:
                    img_bytes = imgp.read_bytes()
                    img = decode_image_bytes(img_bytes)

                    faces, _ = det_adapter.detect(img)
                    if len(faces) == 0:
                        continue

                    best_face = max(faces, key=lambda x: (x.x2 - x.x1) * (x.y2 - x.y1))

                    H, W = img.shape[:2]
                    fx1, fy1, fx2, fy2 = (
                        float(best_face.x1),
                        float(best_face.y1),
                        float(best_face.x2),
                        float(best_face.y2),
                    )
                    bw = max(1.0, fx2 - fx1)
                    bh = max(1.0, fy2 - fy1)

                    pad = 0.10
                    x1 = max(0, int(fx1 - pad * bw))
                    y1 = max(0, int(fy1 - pad * bh))
                    x2 = min(W, int(fx2 + pad * bw))
                    y2 = min(H, int(fy2 + pad * bh))

                    face_crop = img[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    if blur_skip_enable:
                        try:
                            if calc_blur_score(face_crop) < blur_thres:
                                total_stats["blur_skipped"] += 1
                                continue
                        except Exception:
                            pass

                    landmarks = best_face.landmarks
                    lms2 = None
                    if landmarks is not None:
                        arr = np.array(landmarks, dtype=np.float32).reshape(-1, 2)
                        if arr.shape == (5, 2):
                            arr[:, 0] -= x1
                            arr[:, 1] -= y1
                            lms2 = arr

                    emb = reid.embed(face_crop, lms2)

                    person_embs.append(np.asarray(emb, dtype=np.float32))

                except Exception as e:
                    total_stats["errors"] += 1
                    log_enroll(f"Fehler bei {imgp.name}: {e}")

        if len(person_embs) > 0:
            raw_n = len(person_embs)
            embs = np.stack(person_embs, axis=0).astype(np.float32, copy=False)

            if keep_q > 0.0 and len(embs) > 1:
                embs = filter_outliers_to_centroid(embs, keep_quantile=keep_q)

            if dedup_enable and len(embs) > 1:
                embs = dedupe_greedy(embs, sim_th=dedup_th)

            if proto_k and proto_k > 0 and len(embs) > 0:
                k = min(int(proto_k), int(len(embs)))
                if k >= 1:
                    embs = pick_prototypes(embs, k=k)

            kept_n = int(len(embs)) if embs is not None else 0
            if kept_n > 0:
                for e in embs:
                    temp_db.add_person(name, e)
                total_stats["added"] += kept_n
                log_enroll(
                    f"OK: {name} -> {kept_n} Vektoren (raw={raw_n}, dropped={raw_n - kept_n})."
                )
            else:
                log_enroll(
                    f"WARN: {name} -> 0 Vektoren (raw={raw_n}, seen_images={person_seen})."
                )
        else:
            log_enroll(f"WARN: {name} -> 0 Vektoren (seen_images={person_seen}).")

    temp_db.metadata = {
        "model_tag": paths["tag"],
        "model_path": rec_model_path,
        "preprocess": reid.preprocess,
        "created_utc": utc_rfc1123(),
        "format": "obf-faiss-v2",
        "detector_path": det_model_path,
        "device_det": device_det,
        "device_reid": device_reid,
        "enroll_keep_quantile": keep_q,
        "enroll_blur_skip_enable": blur_skip_enable,
        "enroll_blur_var_thres": blur_thres,
        "enroll_dedup_enable": dedup_enable,
        "enroll_dedup_sim": dedup_th,
        "enroll_prototypes_k": proto_k,
    }

    log_enroll("Speichere Datenbank auf Festplatte...")
    temp_db.save()

    if str(target_db_path).lower().endswith(".faiss"):
        log_enroll(
            f">>> FERTIG. DB gespeichert: {target_db_path.name} + {target_db_path.with_suffix('.json').name}"
        )
    else:
        log_enroll(f">>> FERTIG. Binäre DB gespeichert: {target_db_path.name}")

    log_enroll(
        f"    Gesamt: {total_stats['seen']} Bilder, {total_stats['added']} Embeddings, {total_stats['blur_skipped']} blur-skipped, {total_stats['errors']} errors."
    )

    import time

    time.sleep(0.5)

    return total_stats


def enroll_prune_db(cfg: dict[str, Any]) -> dict[str, Any]:
    log_enroll(">>> START: Pruning...")
    paths = _default_db_paths(cfg)
    in_db_path = Path(str(cfg.get("enroll_out_db") or paths["out_db"]))
    out_db_path = Path(str(cfg.get("enroll_prune_out_db") or paths["out_clean"]))

    if not in_db_path.exists():
        if (
            in_db_path.suffix.lower() == ".json"
            and in_db_path.with_suffix(".faiss").exists()
        ):
            pass
        else:
            log_enroll("Abbruch: Input DB fehlt.")
            return {"error": "Input DB missing"}

    sfx = in_db_path.suffix.lower()
    if sfx not in ("", ".faiss", ".json"):
        log_enroll(
            f"Abbruch: Unsupported input DB extension '{in_db_path.suffix}'. Use .faiss/.json."
        )
        return {"error": "Unsupported input DB extension"}

    db_in = FaceDB(str(in_db_path))
    db_in.load()

    names_map = db_in.names or {}
    input_meta = db_in.metadata or {}

    if db_in.index is None or db_in.index.ntotal == 0:
        log_enroll("DB ist leer.")
        return {"status": "empty"}

    try:
        vectors = db_in.index.reconstruct_n(0, db_in.index.ntotal)
    except Exception as e:
        log_enroll(f"DB Read Error (reconstruct_n): {e}")
        return {"error": str(e)}

    if vectors is None or vectors.size == 0:
        log_enroll("DB ist leer.")
        return {"status": "empty"}

    grouped = {}
    for idx, name in names_map.items():
        if idx < len(vectors):
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(vectors[idx])

    out_db = FaceDB(str(out_db_path))
    out_db._create_empty()

    out_db.metadata = input_meta.copy()
    out_db.metadata["pruned_utc"] = utc_rfc1123()

    min_sim = float(cfg.get("enroll_prune_min_sim", 0.3))
    keep_top = int(cfg.get("enroll_prune_keep_top", 20))
    prune_dedup_sim = float(cfg.get("enroll_prune_dedup_sim", 0.0) or 0.0)
    do_prune_dedup = prune_dedup_sim > 0.0
    removed_dedup = 0
    removed_total = 0

    for name, vecs_list in grouped.items():
        arr = np.array(vecs_list, dtype=np.float32)
        if len(arr) == 0:
            continue
        mean = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        sims = np.dot(arr, mean)
        paired = list(zip(sims, arr, strict=False))
        paired.sort(key=lambda x: x[0], reverse=True)
        kept_count = 0
        kept_vecs = []

        for sim, vec in paired:
            if kept_count >= keep_top:
                removed_total += 1
                continue

            if sim < min_sim:
                removed_total += 1
                continue

            if do_prune_dedup and kept_vecs:
                K = np.stack(kept_vecs, axis=0).astype(np.float32, copy=False)
                mx2 = float(np.max(K @ vec.astype(np.float32, copy=False)))
                if mx2 >= prune_dedup_sim:
                    removed_total += 1
                    removed_dedup += 1
                    continue

            out_db.add_person(name, vec)
            kept_vecs.append(vec)
            kept_count += 1

    out_db.save()

    log_enroll(f"Entfernte Ausreißer: {removed_total}")
    if do_prune_dedup:
        log_enroll(f"Entfernte Duplikate (prune): {removed_dedup}")
    return {
        "status": "pruned",
        "removed": removed_total,
        "removed_dedup": removed_dedup,
    }


async def run_enroll_job(
    kind: str, payload: dict[str, Any], restart_after: bool = False
) -> None:
    import app.state as state

    global _ENROLL_PENDING

    if payload:
        with state.STATE_LOCK:
            state.CFG.update(payload)
            try:
                save_config(state.CFG)
            except Exception:
                import logging as _logging

                try:
                    from app.utils import log_exception_throttled
                except Exception:
                    _logging.getLogger(__name__).exception(
                        "Suppressed exception (was 'pass') [throttle helper import failed]"
                    )
                else:
                    log_exception_throttled(
                        _logging.getLogger(__name__),
                        ".\\app\\services\\facedb.py:324",
                        "Suppressed exception (was 'pass')",
                    )

    with ENROLL_LOCK:
        if state.ENROLL_RUNNING:
            _ENROLL_PENDING = True
            log_enroll(
                "Enroll job läuft bereits -> pending gesetzt (wird danach erneut ausgeführt)."
            )
            return
        state.ENROLL_RUNNING = True
        state.ENROLL_LOG.clear()
        _ENROLL_PENDING = False

    try:
        while True:
            try:
                if kind == "build":
                    await asyncio.to_thread(enroll_build_db, state.CFG)
                    if state.CFG.get("enroll_prune_enable", True):
                        await asyncio.to_thread(enroll_prune_db, state.CFG)

                    try:
                        paths = _default_db_paths(state.CFG)
                        prefer_clean = bool(
                            state.CFG.get("enroll_prune_enable", True)
                        ) and (kind in ("build", "prune"))
                        cand = (
                            state.CFG.get("enroll_prune_out_db")
                            if prefer_clean
                            else state.CFG.get("enroll_out_db")
                        )
                        db_path = Path(
                            str(
                                cand
                                or (
                                    paths["out_clean"]
                                    if prefer_clean
                                    else paths["out_db"]
                                )
                            )
                        )

                        if db_path.suffix.lower() == ".json":
                            db_path = db_path.with_suffix(".faiss")
                        elif db_path.suffix == "":
                            db_path = db_path.with_suffix(".faiss")

                        if db_path.exists():
                            with state.STATE_LOCK:
                                state.CFG["rec_db"] = str(db_path)
                                try:
                                    save_config(state.CFG)
                                except Exception as e:
                                    log_enroll(
                                        f"WARN: Failed to save config after enroll job: {e}"
                                    )

                            try:
                                new_db = FaceDB(str(db_path))
                                new_db.load()
                                with state.STATE_LOCK:
                                    if getattr(state, "PIPE", None) is not None:
                                        state.PIPE.facedb = new_db
                                log_enroll(f"Runtime FaceDB gesetzt: {db_path}")
                            except Exception as e:
                                log_enroll(
                                    f"WARN: DB Live-Reload fehlgeschlagen ({db_path}): {e}"
                                )
                        else:
                            log_enroll(
                                f"WARN: Output-DB nicht gefunden, rec_db bleibt unverändert: {db_path}"
                            )
                    except Exception as e:
                        log_enroll(f"WARN: Konnte rec_db nicht automatisch setzen: {e}")

                    log_enroll("Job erfolgreich beendet.")
                elif kind == "prune":
                    await asyncio.to_thread(enroll_prune_db, state.CFG)

                    try:
                        paths = _default_db_paths(state.CFG)
                        prefer_clean = bool(
                            state.CFG.get("enroll_prune_enable", True)
                        ) and (kind in ("build", "prune"))
                        cand = (
                            state.CFG.get("enroll_prune_out_db")
                            if prefer_clean
                            else state.CFG.get("enroll_out_db")
                        )
                        db_path = Path(
                            str(
                                cand
                                or (
                                    paths["out_clean"]
                                    if prefer_clean
                                    else paths["out_db"]
                                )
                            )
                        )

                        if db_path.suffix.lower() == ".json":
                            db_path = db_path.with_suffix(".faiss")
                        elif db_path.suffix == "":
                            db_path = db_path.with_suffix(".faiss")

                        if db_path.exists():
                            with state.STATE_LOCK:
                                state.CFG["rec_db"] = str(db_path)
                                try:
                                    save_config(state.CFG)
                                except Exception as e:
                                    log_enroll(
                                        f"WARN: Failed to save config after enroll job: {e}"
                                    )

                            try:
                                new_db = FaceDB(str(db_path))
                                new_db.load()
                                with state.STATE_LOCK:
                                    if getattr(state, "PIPE", None) is not None:
                                        state.PIPE.facedb = new_db
                                log_enroll(f"Runtime FaceDB gesetzt: {db_path}")
                            except Exception as e:
                                log_enroll(
                                    f"WARN: DB Live-Reload fehlgeschlagen ({db_path}): {e}"
                                )
                        else:
                            log_enroll(
                                f"WARN: Output-DB nicht gefunden, rec_db bleibt unverändert: {db_path}"
                            )
                    except Exception as e:
                        log_enroll(f"WARN: Konnte rec_db nicht automatisch setzen: {e}")

                    log_enroll("Prune Job beendet.")
            except Exception as e:
                log_enroll(f"KRITISCHER FEHLER: {e}")
                import traceback

                log_enroll(traceback.format_exc())
                restart_after = False

            with ENROLL_LOCK:
                if _ENROLL_PENDING:
                    _ENROLL_PENDING = False
                    log_enroll(">>> Neue Änderungen erkannt -> starte Build erneut.")
                    kind = "build"
                    continue
            break

    finally:
        with ENROLL_LOCK:
            state.ENROLL_RUNNING = False
            _ENROLL_PENDING = False

        if restart_after:
            log_enroll(">>> Auto-Restart in 3 Sekunden.")
            await asyncio.sleep(3.0)
            hard_exit(0)
