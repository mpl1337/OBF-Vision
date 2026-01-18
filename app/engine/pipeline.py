import json
import logging
import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openvino import Core as OVCore
from PIL import Image

from app.config import COCO80, PipelineConfig
from app.services.stats import STATS
from app.utils import calc_blur_score, iou_xyxy, nms_xyxy

from .arcface import ArcFaceReID
from .core import GLOBAL_CORE, OVModel
from .face import FaceDB
from .face_aggregator import FaceAggregator
from .face_utils import align_face, check_face_quality
from .tracker import SimpleTracker
from .yolo import YoloOVAdapter
from .yolo_face import FaceDet, YoloFaceAdapter

LOG = logging.getLogger("obf.pipeline")

FILE_IO_QUEUE_MAX = int(os.getenv("OBF_FILEIO_MAX", "2048"))
FILE_IO_QUEUE = queue.Queue(maxsize=FILE_IO_QUEUE_MAX)


def file_io_worker():
    while True:
        task = FILE_IO_QUEUE.get()
        if task is None:
            FILE_IO_QUEUE.task_done()
            break
        func, args = task
        try:
            func(*args)
        except Exception:
            LOG.exception("IO worker task failed")
        finally:
            FILE_IO_QUEUE.task_done()


threading.Thread(target=file_io_worker, daemon=True).start()


class VisionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.core = GLOBAL_CORE if GLOBAL_CORE else OVCore()

        LOG.info("Pipeline Multi-Device Init:")
        LOG.info("  > YOLO Detection: %s", cfg.device_yolo)
        LOG.info("  > Face Detection: %s", cfg.device_face_det)
        LOG.info("  > Face ReID:      %s", cfg.device_reid)

        self.track_lock = threading.Lock()
        self._source_locks = defaultdict(threading.Lock)
        self._locks_lock = threading.Lock()

        hint = getattr(cfg, "perf_hint", "LATENCY")

        if not Path(cfg.yolo_model).exists():
            raise FileNotFoundError(
                f"YOLO model not found: {cfg.yolo_model}. "
                f"Lege .xml/.bin in ./models ab oder nutze den Model-Downloader in der GUI."
            )

        self.yolo_ov = OVModel(
            self.core, cfg.yolo_model, cfg.device_yolo, perf_hint=hint
        )
        self.yolo = YoloOVAdapter(
            self.yolo_ov,
            COCO80,
            conf_thres=cfg.yolo_conf,
            iou_thres=cfg.yolo_iou,
            max_det=cfg.yolo_max_det,
        )

        self.tracking_enable = bool(getattr(cfg, "tracking_enable", True))
        self.keyframe_interval = max(1, int(getattr(cfg, "keyframe_interval", 5)))
        self.track_iou_thres = float(getattr(cfg, "track_iou_thres", 0.5))
        self.track_max_lost = int(getattr(cfg, "track_max_lost", 10))
        self.frame_counters: dict[str, int] = {}
        self.trackers: dict[str, SimpleTracker] = {}
        self._last_face_preds: dict[str, list[dict]] = {}

        self.reid_votes = max(1, int(getattr(cfg, "reid_votes", 3)))
        self.face_agg = FaceAggregator(max_embeds=self.reid_votes * 4)
        self.face_agg_ttl_s = float(os.getenv("OBF_FACE_AGG_TTL_SEC", "90"))

        cpu_cnt = os.cpu_count() or 8
        self._face_exec = ThreadPoolExecutor(max_workers=min(8, max(1, cpu_cnt)))
        self._reid_exec = ThreadPoolExecutor(max_workers=min(4, max(1, cpu_cnt)))

        self.face_det: YoloFaceAdapter | None = None

        if cfg.face_enable:
            preferred = getattr(cfg, "face_det_model", None) or ""
            preferred = str(preferred).strip()

            candidates: list[str] = []
            if preferred:
                candidates.append(preferred)

            candidates += [
                "./models/yolov8n-face-lindeboom.xml",
                "./models/yolov8n-face.xml",
            ]

            face_xml = None
            for c in candidates:
                if c and Path(c).exists():
                    face_xml = c
                    break

            if face_xml:
                LOG.info(
                    "Loading Face Detector on %s: %s", cfg.device_face_det, face_xml
                )
                self.face_det = YoloFaceAdapter(
                    OVModel(self.core, face_xml, cfg.device_face_det, perf_hint=hint),
                    conf_thres=cfg.face_conf,
                )
            else:
                LOG.warning(
                    "Kein Face-Modell für Detection gefunden! (face_det_model='%s')",
                    preferred,
                )

        self.reid: ArcFaceReID | None = None
        self.facedb: FaceDB | None = None

        if cfg.rec_enable:
            rec_xml = cfg.rec_model if cfg.rec_model else "./models/w600k_mbf.xml"
            if Path(rec_xml).exists():
                LOG.info("Loading Face ReID on %s: %s", cfg.device_reid, rec_xml)
                self.reid = ArcFaceReID(
                    OVModel(self.core, rec_xml, cfg.device_reid, perf_hint=hint),
                    preprocess=cfg.rec_preprocess,
                )

                if cfg.rec_db:
                    LOG.info("Loading Face DB: %s", cfg.rec_db)
                    self.facedb = FaceDB(cfg.rec_db)
                    self.facedb.load()
            else:
                LOG.warning("ReID model missing: %s", rec_xml)

        if cfg.unknown_enable and cfg.unknown_dir:
            Path(cfg.unknown_dir).mkdir(parents=True, exist_ok=True)

        self.active_tracks: dict[str, list[dict[str, Any]]] = {}

    def _get_source_lock(self, source_tag: str):
        with self._locks_lock:
            return self._source_locks[source_tag]

    def shutdown(self):
        LOG.info("Pipeline Shutdown: Stoppe Thread-Executors...")

        def _shutdown_exec(ex):
            if ex is None:
                return
            try:
                ex.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                ex.shutdown(wait=True)
            except Exception:
                pass

        if hasattr(self, "_face_exec"):
            _shutdown_exec(self._face_exec)
        if hasattr(self, "_reid_exec"):
            _shutdown_exec(self._reid_exec)

        self.yolo = None
        self.face_det = None
        self.reid = None
        self.facedb = None
        self.trackers.clear()
        LOG.info("Pipeline Shutdown: Done.")

    def _get_tracker(self, source_tag: str):
        if source_tag not in self.trackers:
            if len(self.trackers) >= 50:
                oldest_tag = min(
                    self.trackers.keys(), key=lambda t: self.frame_counters.get(t, 0)
                )

                del self.trackers[oldest_tag]
                self.frame_counters.pop(oldest_tag, None)
                self._last_face_preds.pop(oldest_tag, None)

                LOG.info(
                    "Tracker evicted (LRU): %s (total: %d)",
                    oldest_tag,
                    len(self.trackers),
                )

            self.trackers[source_tag] = SimpleTracker(
                iou_th=self.track_iou_thres,
                max_lost=self.track_max_lost,
            )

        return self.trackers[source_tag]

    def run(
        self, img_bgr: np.ndarray, source_tag: str, snapshot_mode: bool = False
    ) -> tuple[list[dict], dict[str, Any]]:
        from types import SimpleNamespace

        source_lock = self._get_source_lock(source_tag)

        with source_lock:
            with self.track_lock:
                fc = self.frame_counters.get(source_tag, 0) + 1
                self.frame_counters[source_tag] = fc

                tracker = None
                if (not snapshot_mode) and bool(self.tracking_enable):
                    tracker = self._get_tracker(source_tag)

            ydbg: dict[str, Any] = {}
            obj_preds: list[dict[str, Any]] = []
            roi_dets = []

            if snapshot_mode or (not bool(self.tracking_enable)):
                dets, ydbg = self.yolo.detect(img_bgr)

                for d in dets or []:
                    x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                    lbl = str(d.label)
                    obj_preds.append(
                        {
                            "x_min": float(x1),
                            "y_min": float(y1),
                            "x_max": float(x2),
                            "y_max": float(y2),
                            "confidence": float(d.conf),
                            "label": lbl,
                        }
                    )
                    if lbl.lower() == "person":
                        roi_dets.append(
                            SimpleNamespace(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                label="person",
                                track_id=None,
                            )
                        )

                is_keyframe = True

            else:
                is_keyframe = (fc % self.keyframe_interval == 0) or (fc == 1)

                if is_keyframe:
                    dets, ydbg = self.yolo.detect(img_bgr)
                    det_tuples = [
                        (
                            int(d.x1),
                            int(d.y1),
                            int(d.x2),
                            int(d.y2),
                            float(d.conf),
                            str(d.label),
                        )
                        for d in (dets or [])
                    ]
                    tracks = tracker.update(det_tuples) if tracker is not None else []
                else:
                    tracks = tracker.update([]) if tracker is not None else []

                for t in tracks or []:
                    x1, y1, x2, y2 = t.bbox
                    obj_preds.append(
                        {
                            "x_min": float(x1),
                            "y_min": float(y1),
                            "x_max": float(x2),
                            "y_max": float(y2),
                            "confidence": float(t.conf),
                            "label": t.label,
                            "track_id": int(t.id),
                        }
                    )
                    if t.label == "person":
                        roi_dets.append(
                            SimpleNamespace(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                label="person",
                                track_id=int(t.id),
                            )
                        )

            fdbg: dict[str, Any] = {}
            if is_keyframe and self.cfg.face_enable and self.face_det:
                dets_for_roi = roi_dets if self.cfg.face_roi else None
                face_preds, fdbg_list = self._process_faces(
                    img_bgr, dets_for_roi, source_tag
                )

                with self.track_lock:
                    self._last_face_preds[source_tag] = face_preds

                if fdbg_list:
                    fdbg = fdbg_list[0]

            face_preds = self._last_face_preds.get(source_tag, [])
            preds = obj_preds + face_preds

            return preds, {
                "yolo": ydbg,
                "faces": fdbg,
                "keyframe": bool(is_keyframe),
                "frame": int(fc),
                "source": source_tag,
                "keyframe_interval": int(self.keyframe_interval),
                "tracking": bool(self.tracking_enable) and (not snapshot_mode),
                "snapshot_mode": bool(snapshot_mode),
            }

    def _process_faces(
        self, img_bgr: np.ndarray, dets, source_tag: str
    ) -> tuple[list[dict], list[dict]]:
        H, W = img_bgr.shape[:2]
        rois = []

        if dets is not None:
            for d in dets:
                if d.label == "person":
                    w_box = d.x2 - d.x1
                    h_box = d.y2 - d.y1
                    pad_x, pad_y = 0.20 * w_box, 0.20 * h_box
                    x1 = int(max(0, d.x1 - pad_x))
                    y1 = int(max(0, d.y1 - pad_y))
                    x2 = int(min(W - 1, d.x2 + pad_x))
                    y2 = int(min(H - 1, d.y2 + pad_y))
                    rois.append((x1, y1, x2, y2, getattr(d, "track_id", None)))

            if not rois:
                with self.track_lock:
                    self.active_tracks[source_tag] = []
                return [], []
        else:
            rois = [(0, 0, W - 1, H - 1, None)]

        crops, valid_rois = [], []
        for r in rois:
            c = img_bgr[r[1] : r[3], r[0] : r[2]]
            if c.size > 0:
                crops.append(c)
                valid_rois.append(r)

        if not crops:
            return [], []

        face_results: list[tuple[list[FaceDet], dict[str, Any]]] = [
            ([], {"info": "init"}) for _ in range(len(crops))
        ]

        def detect_job(idx, crop):
            return idx, self.face_det.detect(crop, min_px=int(self.cfg.face_min_px))

        if len(crops) == 1:
            i, res = detect_job(0, crops[0])
            face_results[i] = res
        else:
            future_to_idx = {
                self._face_exec.submit(detect_job, i, c): i for i, c in enumerate(crops)
            }
            for fut, idx in future_to_idx.items():
                try:
                    _i, res = fut.result(timeout=2.0)

                    face_results[idx] = res
                except FutTimeoutError:
                    LOG.warning(
                        "Face detect future timeout (2.0s) – skipping one ROI crop"
                    )
                    face_results[idx] = ([], {"error": "face_detect_timeout"})
                except Exception:
                    LOG.exception("Face detect future failed – skipping one ROI crop")
                    face_results[idx] = ([], {"error": "face_detect_failed"})

        with self.track_lock:
            prev_tracks = list(self.active_tracks.get(source_tag, []))

        current_tracks: list[dict[str, Any]] = []
        out_faces_raw: list[dict[str, Any]] = []
        dbg_list: list[dict[str, Any]] = []
        reid_queue = []

        IOU_THRESH, MAX_SKIP_FRAMES = 0.45, 8

        for i, (faces_raw, dbg) in enumerate(face_results):
            dbg_list.append(dbg)
            rx1, ry1, _, _, owner_id = valid_rois[i]
            crop = crops[i]

            for f in faces_raw:
                fx1, fy1, fx2, fy2 = f.x1 + rx1, f.y1 + ry1, f.x2 + rx1, f.y2 + ry1
                current_box = [fx1, fy1, fx2, fy2]

                label, track_skip, needs_reid = "face", 0, False
                matched_track, best_iou = None, 0.0

                for trk in prev_tracks:
                    iou = iou_xyxy(current_box, trk["box"])
                    if iou > best_iou:
                        best_iou = iou
                        matched_track = trk

                if (
                    matched_track
                    and best_iou > IOU_THRESH
                    and matched_track["skip"] < MAX_SKIP_FRAMES
                ):
                    label = matched_track["label"]
                    track_skip = matched_track["skip"] + 1
                else:
                    if self.cfg.rec_enable and self.reid:
                        needs_reid = True
                    else:
                        label, track_skip = "face", 0

                        if self.cfg.unknown_enable and self.cfg.unknown_dir:
                            lx1, ly1, lx2, ly2 = map(int, (f.x1, f.y1, f.x2, f.y2))
                            h, w = crop.shape[:2]

                            lx1 = max(0, min(w, lx1))
                            lx2 = max(0, min(w, lx2))
                            ly1 = max(0, min(h, ly1))
                            ly2 = max(0, min(h, ly2))

                            if lx2 > lx1 and ly2 > ly1:
                                face_crop = crop[ly1:ly2, lx1:lx2]
                                self._save_unknown(
                                    face_crop, 0.0, source_tag, suffix="norec", emb=None
                                )

                bad_names = {"Unknown", "Blurry", "PoorQuality", "face"}
                userid = None
                if label not in bad_names:
                    userid = str(label)

                face_entry = {
                    "x_min": float(fx1),
                    "y_min": float(fy1),
                    "x_max": float(fx2),
                    "y_max": float(fy2),
                    "confidence": float(f.conf),
                    "label": label,
                    "userid": userid,
                    "_box": current_box,
                    "_skip": track_skip,
                    "_owner": owner_id,
                    "_score": 0.0,
                    "_emb": None,
                }
                idx_in_list = len(out_faces_raw)
                out_faces_raw.append(face_entry)
                if needs_reid:
                    reid_queue.append((idx_in_list, crop, f, (rx1, ry1), owner_id))

        if reid_queue:

            def reid_job(q_idx, crop_img, f_det, offset, owner_id):
                name, score, emb = self._identify_arcface(
                    img_bgr, offset, crop_img, f_det, source_tag
                )
                return q_idx, owner_id, name, score, emb

            if len(reid_queue) == 1:
                idx, c, fd, off, owner_id = reid_queue[0]
                _idx, _owner, name, score, emb = reid_job(idx, c, fd, off, owner_id)
                out_faces_raw[_idx]["label"] = name
                bad_names = {"Unknown", "Blurry", "PoorQuality", "face"}
                out_faces_raw[_idx]["userid"] = None if name in bad_names else str(name)
                out_faces_raw[_idx]["_score"] = float(score)
                out_faces_raw[_idx]["_emb"] = emb
                out_faces_raw[_idx]["_owner"] = _owner
            else:
                futures = [
                    self._reid_exec.submit(reid_job, idx, c, fd, off, owner_id)
                    for idx, c, fd, off, owner_id in reid_queue
                ]
                for fut in futures:
                    try:
                        _idx, _owner, name, score, emb = fut.result(timeout=2.0)
                    except FutTimeoutError:
                        LOG.warning("ReID future timeout (2.0s) – keeping label as-is")
                        continue
                    except Exception:
                        LOG.exception("ReID future failed – keeping label as-is")
                        continue

                    out_faces_raw[_idx]["label"] = name
                    bad_names = {"Unknown", "Blurry", "PoorQuality", "face"}
                    out_faces_raw[_idx]["userid"] = (
                        None if name in bad_names else str(name)
                    )
                    out_faces_raw[_idx]["_score"] = float(score)
                    out_faces_raw[_idx]["_emb"] = emb
                    out_faces_raw[_idx]["_owner"] = _owner

            if self.cfg.rec_enable and self.reid and self.facedb:
                try:
                    self.face_agg.prune(self.face_agg_ttl_s)
                except Exception:
                    LOG.exception("FaceAggregator prune failed")

                for fe in out_faces_raw:
                    owner = fe.get("_owner", None)
                    if owner is None:
                        continue
                    key = (source_tag, owner)
                    emb = fe.get("_emb", None)
                    if emb is None:
                        continue
                    self.face_agg.add(key, emb, 1.0)
                    if self.face_agg.count(key) >= self.reid_votes:
                        mean_emb = self.face_agg.mean(key)
                        if mean_emb is None:
                            continue
                        m2 = self.facedb.best_match(mean_emb)
                        if m2 is not None and float(m2.score) >= float(
                            self.cfg.rec_thres
                        ):
                            fe["label"] = m2.name
                            fe["userid"] = str(m2.name)

        drop_labels = {"Blurry", "PoorQuality"}

        filtered_faces = []
        for fe in out_faces_raw:
            lbl = str(fe.get("label", "") or "")

            if lbl in drop_labels:
                continue

            if lbl.strip().lower() == "face" or lbl.strip() == "":
                fe["label"] = "Unknown"
                fe["userid"] = None

            if str(fe.get("label", "") or "").strip().lower() == "unknown":
                fe["userid"] = None

            filtered_faces.append(fe)

        out_faces_raw = filtered_faces

        dbg_enabled = LOG.isEnabledFor(logging.DEBUG)
        dbg_lines = 0
        dbg_limit = 20
        dropped = 0

        def _dbg(msg, *args):
            nonlocal dbg_lines
            if not dbg_enabled:
                return
            if dbg_lines >= dbg_limit:
                return
            LOG.debug(msg, *args)
            dbg_lines += 1

        def _area(b):
            return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

        def _ios(a, b):
            x1 = max(float(a[0]), float(b[0]))
            y1 = max(float(a[1]), float(b[1]))
            x2 = min(float(a[2]), float(b[2]))
            y2 = min(float(a[3]), float(b[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if inter <= 0.0:
                return 0.0
            aa = _area(a)
            ab = _area(b)
            m = min(aa, ab)
            return 0.0 if m <= 1e-9 else float(inter / m)

        def _dedupe_nested(entries, ios_th=0.90):
            nonlocal dropped
            if len(entries) <= 1:
                return entries

            def _key(e):
                b = e.get("_box", (0, 0, 0, 0))
                area = _area(b)
                conf = float(e.get("confidence", 0.0) or 0.0)
                has_uid = 1 if (e.get("userid") not in (None, "", "null")) else 0
                return (has_uid, conf, area)

            sorted_e = sorted(entries, key=_key, reverse=True)
            kept = []
            for e in sorted_e:
                b = e.get("_box", None)
                if b is None:
                    kept.append(e)
                    continue

                is_dup = False
                for k in kept:
                    kb = k.get("_box", None)
                    if kb is None:
                        continue

                    ios_v = _ios(b, kb)
                    if ios_v >= ios_th:
                        is_dup = True
                        dropped += 1
                        _dbg(
                            "[NESTED] drop box=%s conf=%.3f uid=%s nested_in=%s keep_conf=%.3f keep_uid=%s ios=%.3f th=%.2f",
                            tuple(map(int, b)),
                            float(e.get("confidence", 0.0) or 0.0),
                            e.get("userid"),
                            tuple(map(int, kb)),
                            float(k.get("confidence", 0.0) or 0.0),
                            k.get("userid"),
                            ios_v,
                            ios_th,
                        )
                        break

                if not is_dup:
                    kept.append(e)

            return kept

        by_owner: dict[Any, list[dict[str, Any]]] = {}
        for e in out_faces_raw:
            by_owner.setdefault(e.get("_owner", None), []).append(e)

        tmp = []
        for owner, lst in by_owner.items():
            before = len(lst)
            lst2 = _dedupe_nested(lst, ios_th=0.90)
            after = len(lst2)
            if dbg_enabled and before != after:
                _dbg("[NESTED] owner=%s before=%d after=%d", owner, before, after)
            tmp.extend(lst2)

        before_all = len(tmp)
        out_faces_raw = _dedupe_nested(tmp, ios_th=0.90)
        after_all = len(out_faces_raw)

        if dbg_enabled:
            LOG.debug(
                "[NESTED] summary dropped=%d owners=%d before_all=%d after_all=%d (printed_lines=%d/%d)",
                dropped,
                len(by_owner),
                before_all,
                after_all,
                dbg_lines,
                dbg_limit,
            )

        final_faces: list[dict[str, Any]] = []

        if len(out_faces_raw) > 0:
            boxes = np.array([f["_box"] for f in out_faces_raw], dtype=np.float32)
            scores = np.array(
                [f["confidence"] for f in out_faces_raw], dtype=np.float32
            )
            keep_indices = nms_xyxy(boxes, scores, iou_th=0.30)
            for i in keep_indices:
                final_faces.append(out_faces_raw[i])

        for entry in final_faces:
            lbl = entry["label"]
            STATS.update(lbl)
            current_tracks.append(
                {"box": entry["_box"], "label": lbl, "skip": entry["_skip"]}
            )
            del entry["_box"], entry["_skip"]

        with self.track_lock:
            self.active_tracks[source_tag] = current_tracks

        return final_faces, dbg_list

    def _identify_arcface(
        self,
        full_img: np.ndarray,
        offset: tuple,
        crop_bgr: np.ndarray,
        face_det: FaceDet,
        source_tag: str,
    ) -> tuple[str, float, np.ndarray | None]:
        fx1, fy1, fx2, fy2 = (
            int(face_det.x1),
            int(face_det.y1),
            int(face_det.x2),
            int(face_det.y2),
        )
        h_img, w_img = crop_bgr.shape[:2]
        w_face, h_face = fx2 - fx1, fy2 - fy1
        pad = 0.15
        sx1, sy1 = max(0, int(fx1 - pad * w_face)), max(0, int(fy1 - pad * h_face))
        sx2, sy2 = (
            min(w_img, int(fx2 + pad * w_face)),
            min(h_img, int(fy2 + pad * h_face)),
        )
        raw_face_img = crop_bgr[sy1:sy2, sx1:sx2]
        save_img, img_type_suffix, emb = raw_face_img, "raw", None

        if self.cfg.rec_align:
            if face_det.landmarks is not None and face_det.landmarks.size >= 10:
                lm_local = np.array(face_det.landmarks).reshape(-1, 2)
                rx1, ry1 = offset
                lms_global = lm_local + np.array([rx1, ry1])
                w_target, h_target = self.reid.input_size
                aligned_face = align_face(
                    full_img, lms_global, out_size=(w_target, h_target)
                )
                if aligned_face is not None and aligned_face.size > 0:
                    save_img, img_type_suffix = aligned_face, "ali"
                    emb = self.reid.embed(aligned_face, None)

        if emb is None:
            if raw_face_img.size == 0:
                return "Unknown", 0.0, None
            emb = self.reid.embed(raw_face_img, None)

        if self.cfg.face_quality_enable:
            q_score = check_face_quality(face_det.landmarks)
            if q_score < self.cfg.face_quality_thres:
                self._save_unknown(
                    save_img, 0.0, source_tag, suffix=f"badqual_{q_score:.2f}"
                )
                return "PoorQuality", 0.0, None

        if self.cfg.rec_blur_enable:
            try:
                b = float(calc_blur_score(save_img))
            except Exception:
                b = 1e9

            if b < float(self.cfg.rec_blur_thres):
                try:
                    self._save_unknown(
                        save_img, 0.0, source_tag, suffix=f"blurry_{b:.1f}", emb=None
                    )
                except Exception:
                    pass
                return "Unknown", 0.0, None

        if not self.facedb:
            self._save_unknown(
                save_img, 0.0, source_tag, suffix=img_type_suffix, emb=emb
            )
            return "Unknown", 0.0, emb

        match = self.facedb.best_match(emb)
        score = float(match.score) if match is not None else 0.0

        if match is None or score < float(self.cfg.rec_thres):
            self._save_unknown(
                save_img, score, source_tag, suffix=img_type_suffix, emb=emb
            )
            return "Unknown", score, emb

        return match.name, score, emb

    def _save_unknown(
        self,
        face_bgr: np.ndarray,
        score: float,
        source_tag: str,
        suffix: str = "raw",
        emb: np.ndarray | None = None,
    ):
        if (
            not self.cfg.unknown_enable
            or not self.cfg.unknown_dir
            or face_bgr is None
            or face_bgr.size == 0
        ):
            return

        img_copy = face_bgr.copy()

        def _do_write(img, s, src, suf, e):
            try:
                t_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fn = f"{t_str}_{src}_{suf}_sim{s:.3f}.jpg"
                p = Path(self.cfg.unknown_dir) / fn

                ok = cv2.imwrite(str(p), img)
                if not ok:
                    Image.fromarray(img[:, :, ::-1]).save(p)

                if e is not None:
                    np.save(str(p.with_suffix(".npy")), e.astype(np.float32))

                meta = {"score": float(s), "source": str(src), "suffix": str(suf)}
                p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
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
                        ".\\app\\engine\\pipeline.py:_save_unknown",
                        "Suppressed exception (was 'pass')",
                    )

        try:
            FILE_IO_QUEUE.put_nowait(
                (_do_write, (img_copy, score, source_tag, suffix, emb))
            )
        except queue.Full:
            LOG.warning(
                "FILE_IO_QUEUE full (%s) - dropping unknown face save",
                FILE_IO_QUEUE_MAX,
            )


def _pick_db_from_cfg(cfg_dict: dict[str, Any]) -> str | None:
    """Pick a FaceDB path for runtime. Supports .faiss (with adjacent .json metadata) or selecting the .json."""
    p = cfg_dict.get("rec_db")
    if p:
        db_path = Path(str(p))

        if db_path.suffix.lower() == ".json":
            db_path = db_path.with_suffix(".faiss")
        elif db_path.suffix == "":
            db_path = db_path.with_suffix(".faiss")

        if db_path.exists() and db_path.suffix.lower() == ".faiss":
            return str(db_path)

        try:
            if db_path.suffix and db_path.suffix.lower() != ".faiss":
                LOG.warning(
                    "rec_db has unsupported extension (%s). Use .faiss or .json.",
                    db_path.suffix,
                )
            elif not db_path.exists():
                LOG.warning("rec_db file missing: %s", db_path)
        except Exception:
            pass

    for f in [Path("./faces_db_clean.faiss"), Path("./faces_db.faiss")]:
        if f.exists():
            return str(f)

    return None


def build_pipeline_from_cfg(cfg_dict: dict[str, Any]) -> VisionPipeline:
    global_dev = str(cfg_dict.get("device", "CPU"))

    try:
        cap = int(cfg_dict.get("ov_pool_cap", 32) or 32)
    except Exception:
        cap = 32
    try:
        mn = int(cfg_dict.get("ov_pool_min", 1) or 1)
    except Exception:
        mn = 1

    if cap < 1:
        cap = 1
    if mn < 1:
        mn = 1
    if mn > cap:
        cap = mn

    os.environ["OBF_OV_POOL_CAP"] = str(cap)
    os.environ["OBF_OV_POOL_MIN"] = str(mn)

    LOG.info("OV Pool bounds: OBF_OV_POOL_MIN=%s, OBF_OV_POOL_CAP=%s", mn, cap)

    face_det_model = str(cfg_dict.get("face_det_model", "") or "").strip()
    if not face_det_model:
        face_det_model = None

    pcfg = PipelineConfig(
        device_yolo=str(cfg_dict.get("device_yolo", global_dev)),
        device_face_det=str(cfg_dict.get("device_face_det", global_dev)),
        device_reid=str(cfg_dict.get("device_reid", global_dev)),
        perf_hint=str(cfg_dict.get("perf_hint", "LATENCY")),
        yolo_model=str(cfg_dict.get("yolo_model", "./models/yolo11s.xml")),
        yolo_conf=float(cfg_dict.get("yolo_conf", 0.35)),
        yolo_iou=float(cfg_dict.get("yolo_iou", 0.50)),
        yolo_max_det=int(cfg_dict.get("yolo_max_det", 100)),
        face_enable=bool(cfg_dict.get("face_enable", False)),
        face_roi=bool(cfg_dict.get("face_roi", True)),
        face_conf=float(cfg_dict.get("face_conf", 0.45)),
        face_min_conf=float(cfg_dict.get("face_min_conf", 0.30)),
        face_min_px=int(cfg_dict.get("face_min_px", 32)),
        face_det_model=face_det_model,
        face_quality_enable=bool(cfg_dict.get("face_quality_enable", False)),
        face_quality_thres=float(cfg_dict.get("face_quality_thres", 0.4)),
        rec_enable=bool(cfg_dict.get("rec_enable", True)),
        rec_align=bool(cfg_dict.get("rec_align", True)),
        rec_model=cfg_dict.get("rec_model"),
        rec_db=_pick_db_from_cfg(cfg_dict),
        rec_thres=float(cfg_dict.get("rec_thres", 0.55)),
        rec_preprocess=str(cfg_dict.get("rec_preprocess", "auto")),
        rec_blur_enable=bool(cfg_dict.get("rec_blur_enable", True)),
        rec_blur_thres=float(cfg_dict.get("rec_blur_thres", 60.0)),
        unknown_enable=bool(cfg_dict.get("unknown_enable", True)),
        unknown_dir=str(cfg_dict.get("unknown_dir", "./unknown_faces")),
        tracking_enable=bool(cfg_dict.get("tracking_enable", True)),
        keyframe_interval=int(cfg_dict.get("keyframe_interval", 5)),
        track_iou_thres=float(cfg_dict.get("track_iou_thres", 0.5)),
        track_max_lost=int(cfg_dict.get("track_max_lost", 10)),
        reid_votes=int(cfg_dict.get("reid_votes", 3)),
    )
    return VisionPipeline(pcfg)


def ensure_pipeline_loaded():
    import app.state as state

    with state.STATE_LOCK:
        if state.PIPE is None:
            state.PIPE = build_pipeline_from_cfg(state.CFG)
