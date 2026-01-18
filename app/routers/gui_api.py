import asyncio
import json
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse

import app.state as state
from app.config import GUI_SCHEMA, save_config
from app.engine.core import GLOBAL_CORE
from app.engine.pipeline import ensure_pipeline_loaded
from app.schemas import (
    ModelJobPayload,
    PersonFilePayload,
    PersonPayload,
    PipelineTogglePayload,
    UnknownAssignPayload,
)
from app.services.facedb import run_enroll_job
from app.services.models import (
    MODEL_CATALOG,
    delete_model_files,
    get_models_dir,
    get_models_storage_info,
    run_model_job,
)
from app.services.stats import STATS
from app.services.unknown_cluster import UnknownClusterService
from app.state import (
    CFG,
    ENROLL_LOCK,
    ENROLL_LOG,
    MODEL_LOCK,
    MODEL_LOG,
    PERF,
    START_TIME,
    STATE_LOCK,
    WS_MANAGER,
)
from app.utils import LOG, hard_exit, norm_ov_device_name

router = APIRouter()


def _try_bind(host: str, port: int) -> tuple[bool, str]:
    h = (host or "").strip() or "0.0.0.0"
    is_v6 = (":" in h) and (h.count(":") >= 2)

    s = None
    try:
        s = socket.socket(
            socket.AF_INET6 if is_v6 else socket.AF_INET, socket.SOCK_STREAM
        )

        if sys.platform.startswith("win"):
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except Exception:
                pass
        else:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except Exception:
                pass

        if is_v6:
            s.bind((h, port, 0, 0))
        else:
            s.bind((h, port))

        s.listen(1)
        return True, "OK"
    except OSError as e:
        return False, str(e)
    finally:
        try:
            if s:
                s.close()
        except Exception:
            pass


@router.get("/gui/port/check")
async def gui_port_check(
    port: int = Query(..., ge=1, le=65535),
    host: str = "0.0.0.0",
):
    with STATE_LOCK:
        cur_host = str(CFG.get("host", "0.0.0.0"))
        cur_port = int(CFG.get("port", 32168))

    if int(port) == int(cur_port):
        return {
            "status": "warn",
            "available": True,
            "message": f"Port {port} ist aktuell vom laufenden Server belegt",
            "host": host,
            "port": port,
        }

    ok, info = _try_bind(host, int(port))
    if ok:
        return {
            "status": "ok",
            "available": True,
            "message": f"Port {port} ist frei.",
            "host": host,
            "port": port,
        }

    return {
        "status": "error",
        "available": False,
        "message": f"Port {port} ist belegt oder nicht bindbar: {info}",
        "host": host,
        "port": port,
    }


def safe_join(base: Path, *parts: str) -> Path:
    if not parts:
        raise ValueError("No path parts provided")

    base_resolved = base.resolve()
    target = base_resolved.joinpath(*parts).resolve()

    try:
        target.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path traversal blocked: {parts}") from None

    return target


async def broadcast_status_loop():
    """Hintergrund-Task: Sendet Status & Logs via WS"""
    last_model_log_len = 0
    last_enroll_log_len = 0

    while True:
        try:
            await asyncio.sleep(1.0)

            with WS_MANAGER.lock:
                has_conns = bool(WS_MANAGER.active_connections)
            if not has_conns:
                continue

            uptime_sec = int(time.time() - START_TIME)
            d, rem = divmod(uptime_sec, 86400)
            h, rem = divmod(rem, 3600)
            m, s = divmod(rem, 60)
            uptime = f"{d}d {h:02}:{m:02}:{s:02}" if d > 0 else f"{h:02}:{m:02}:{s:02}"

            with STATE_LOCK:
                pipe_enabled = bool(state.PIPE_ENABLED)

            perf = PERF.snapshot()

            ema_ms = float(perf.get("ema_ms", 0.0) or 0.0)
            last_ms = float(perf.get("last_ms", 0.0) or 0.0)
            fps = (1000.0 / ema_ms) if ema_ms > 0 else 0.0

            status_msg = {
                "type": "status",
                "data": {
                    "pipeline_enabled": pipe_enabled,
                    "uptime": uptime,
                    "req_count": int(perf.get("req_count", 0)),
                    "ema_ms": round(ema_ms, 1),
                    "last_ms": round(last_ms, 1),
                    "fps": round(fps, 1),
                    "last_labels": (perf.get("last_labels") or []),
                },
            }

            await WS_MANAGER.broadcast(json.dumps(status_msg))

            with MODEL_LOCK:
                model_lines = list(MODEL_LOG)
            curr_ml_len = len(model_lines)
            if curr_ml_len != last_model_log_len:
                await WS_MANAGER.broadcast(
                    json.dumps({"type": "model_log", "lines": model_lines})
                )
                last_model_log_len = curr_ml_len

            with ENROLL_LOCK:
                enroll_lines = list(ENROLL_LOG)
            curr_el_len = len(enroll_lines)
            if curr_el_len != last_enroll_log_len:
                await WS_MANAGER.broadcast(
                    json.dumps({"type": "enroll_log", "lines": enroll_lines})
                )
                last_enroll_log_len = curr_el_len

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"WS Broadcast Error: {e}")
            await asyncio.sleep(1.0)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await WS_MANAGER.connect(websocket)
    try:
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)
    finally:
        WS_MANAGER.disconnect(websocket)


def trigger_hard_restart_thread(delay=0.5):
    def _kill_me():
        LOG.info(f"Neustart in {delay}s angefordert.")
        time.sleep(delay)
        LOG.info("GUI Trigger: Führe hard_exit(42) aus.")
        try:
            hard_exit(42)
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
                    ".\\app\\routers\\gui_api.py:trigger_hard_restart_thread",
                    "Suppressed exception (hard_exit failed)",
                )

    threading.Thread(target=_kill_me, daemon=True).start()


@router.post("/gui/state")
async def set_state(request: Request):
    """
    Erwartet:
      { "config": { ... }, "action": "save" | "apply" }

    - save: nur persistieren
    - apply: persistieren + Neustart via watchdog triggern
    """
    try:
        pl = await request.json()
    except Exception:
        return Response(status_code=400)

    cfg_in = pl.get("config", None)
    action = str(pl.get("action", "save") or "save").strip().lower()

    if not isinstance(cfg_in, dict):
        return Response(status_code=400)

    with STATE_LOCK:
        CFG.update(cfg_in)

        desired_enabled = bool(CFG.get("pipeline_enabled", True))
        state.PIPE_ENABLED = desired_enabled
        CFG["pipeline_enabled"] = desired_enabled

        try:
            save_config(CFG)
        except Exception:
            LOG.exception("Failed to save config from /gui/state")

    try:
        if desired_enabled:
            ensure_pipeline_loaded()
        else:
            state.stop_pipeline()
    except Exception:
        LOG.exception("Failed to apply pipeline_enabled change")

    if action == "apply":
        trigger_hard_restart_thread(0.5)

    return {"ok": True}


@router.post("/gui/restart")
async def restart_server():
    """Restart button in GUI."""
    trigger_hard_restart_thread(0.5)
    return {"ok": True}


@router.get("/gui/state")
async def get_state():
    schema = GUI_SCHEMA.copy()
    md = get_models_dir()

    def mpath(name: str) -> str:
        return f"./models/{name}"

    yolo_files = sorted(
        [
            mpath(p.name)
            for p in md.glob("*.xml")
            if ("yolo" in p.name.lower() and "face" not in p.name.lower())
        ]
    )
    if not yolo_files:
        yolo_files = ["./models/yolo11s.xml"]
    if "yolo_model" in schema:
        schema["yolo_model"]["options"] = yolo_files

    face_det_files = []

    try:
        for _mid, info in MODEL_CATALOG.items():
            if str(info.get("type", "")).lower() == "face_det":
                fn = info.get("file")
                if fn and (md / fn).exists():
                    face_det_files.append(mpath(fn))
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
                ".\\app\\routers\\gui_api.py:159",
                "Suppressed exception (was 'pass')",
            )

    if not face_det_files:
        face_det_files = sorted(
            [
                mpath(p.name)
                for p in md.glob("*.xml")
                if ("face-detection" in p.name.lower())
                or ("yolo" in p.name.lower() and "face" in p.name.lower())
            ]
        )

    if not face_det_files:
        face_det_files = ["./models/yolov8n-face.xml"]

    if "face_det_model" in schema:
        schema["face_det_model"]["options"] = sorted(list(set(face_det_files)))

    rec_files = []

    try:
        for _mid, info in MODEL_CATALOG.items():
            if str(info.get("type", "")).lower() == "reid":
                fn = info.get("file")
                if fn and (md / fn).exists():
                    rec_files.append(mpath(fn))
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
                ".\\app\\routers\\gui_api.py:185",
                "Suppressed exception (was 'pass')",
            )

    if not rec_files:

        def is_reid_name(n: str) -> bool:
            nl = n.lower()
            if "yolo" in nl and "face" in nl:
                return False
            if "face-detection" in nl:
                return False
            return (
                "reidentification" in nl
                or "arcface" in nl
                or "mbf" in nl
                or "face-recognition" in nl
                or "resnet100" in nl
            )

        rec_files = sorted(
            [mpath(p.name) for p in md.glob("*.xml") if is_reid_name(p.name)]
        )

    if not rec_files:
        rec_files = ["./models/w600k_mbf.xml"]

    if "rec_model" in schema:
        schema["rec_model"]["options"] = sorted(list(set(rec_files)))

    root_p = Path(".")
    db_candidates = []
    try:
        for f in root_p.glob("*.faiss"):
            db_candidates.append(f.name)
        for f in root_p.glob("*.json"):
            db_candidates.append(f.name)
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
                ".\\app\\routers\\gui_api.py:220",
                "Suppressed exception (was 'pass')",
            )

    if not db_candidates:
        db_candidates.append("faces_db_clean.faiss")
        db_candidates.append("faces_db.faiss")

    curr_db = CFG.get("rec_db", "faces_db_clean.faiss")
    if curr_db and curr_db not in db_candidates:
        db_candidates.append(curr_db)

    db_candidates = sorted(list(set(db_candidates)))
    if "rec_db" in schema:
        schema["rec_db"]["options"] = db_candidates

    if GLOBAL_CORE:
        try:
            avail = sorted(list(GLOBAL_CORE.available_devices))
            if "CPU" not in avail:
                avail.insert(0, "CPU")
            for k in ["device", "device_yolo", "device_face_det", "device_reid"]:
                if k in schema:
                    schema[k]["options"] = avail
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
                    ".\\app\\routers\\gui_api.py:243",
                    "Suppressed exception (was 'pass')",
                )

    return {"config": CFG, "schema": schema}


@router.get("/gui/models")
async def get_models_info():
    with STATE_LOCK:
        cfg = dict(CFG)
    md = get_models_dir()
    existing_files = set(p.name for p in md.glob("*"))
    catalog_list = []

    for _mid, info in MODEL_CATALOG.items():
        xml_name = info["file"]
        is_there = xml_name in existing_files
        size = 0
        base = xml_name.replace(".xml", "")
        for ext in [".xml", ".bin", ".onnx", ".pt"]:
            if f"{base}{ext}" in existing_files:
                size += (md / f"{base}{ext}").stat().st_size
        catalog_list.append(
            {
                "id": _mid,
                "name": _mid,
                "desc": info["desc"],
                "type": info["type"],
                "filename": xml_name,
                "exists": is_there,
                "size_mb": round(size / (1024 * 1024), 2),
            }
        )

    yolo_p = cfg.get("yolo_model", "")
    rec_p = cfg.get("rec_model", "")
    db_p = cfg.get("rec_db", "")
    face_det_p = str(cfg.get("face_det_model", "") or "").strip()

    if not face_det_p:
        for c in [
            "./models/yolov8n-face-lindeboom.xml",
            "./models/yolov8n-face.xml",
        ]:
            if Path(c).exists():
                face_det_p = c
                break

    return {
        "catalog": catalog_list,
        "storage": get_models_storage_info(),
        "yolo_model": yolo_p,
        "face_det_model": face_det_p,
        "reid_model": rec_p,
        "db_path": db_p,
        "yolo_exists": bool(yolo_p and Path(yolo_p).exists()),
        "face_det_exists": bool(face_det_p and Path(face_det_p).exists()),
        "reid_exists": bool(rec_p and Path(rec_p).exists()),
        "db_exists": bool(db_p and Path(db_p).exists()),
    }


@router.get("/gui/runtime/models")
async def get_runtime_models():
    with STATE_LOCK:
        cfg = dict(CFG)
        pipe = getattr(state, "PIPE", None)
        enabled = bool(getattr(state, "PIPE_ENABLED", False))

    if not pipe:
        return {
            "loaded": False,
            "pipeline_enabled": enabled,
            "ov_pool_min": cfg.get("ov_pool_min"),
            "ov_pool_cap": cfg.get("ov_pool_cap"),
            "models": [],
        }

    models = []

    def _add(role: str, ov):
        if ov is None:
            return
        try:
            if hasattr(ov, "get_pool_info"):
                info = ov.get_pool_info()
            else:
                info = {"model": None, "device": None, "hint": None}
            info["role"] = role
            models.append(info)
        except Exception as e:
            models.append({"role": role, "error": str(e)})

    _add("YOLO", getattr(pipe, "yolo_ov", None))

    fd = getattr(pipe, "face_det", None)
    _add("FaceDet", getattr(fd, "ov", None) if fd else None)

    reid = getattr(pipe, "reid", None)
    _add("ReID", getattr(reid, "ov", None) if reid else None)

    return {
        "loaded": True,
        "pipeline_enabled": enabled,
        "ov_pool_min": cfg.get("ov_pool_min"),
        "ov_pool_cap": cfg.get("ov_pool_cap"),
        "models": models,
    }


@router.post("/gui/model/start")
async def start_model_job(payload: ModelJobPayload):
    asyncio.create_task(run_model_job(payload.kind, payload.model_dump()))
    return {"ok": True}


@router.post("/gui/model/delete")
async def delete_model_endpoint(request: Request):
    payload = await request.json()
    success = delete_model_files(payload.get("id"))
    return {"ok": success}


@router.get("/gui/model/log")
async def get_model_log():
    with MODEL_LOCK:
        return {"lines": list(MODEL_LOG)}


@router.post("/gui/model/clear")
async def clear_model_log_ep():
    with MODEL_LOCK:
        MODEL_LOG.clear()
    return {"ok": True}


@router.post("/gui/benchmark/run")
async def api_run_benchmark(payload: dict, _background_tasks: BackgroundTasks):
    import openvino as ov

    import app.state as state
    from app.services.models import run_full_matrix_benchmark

    with state.MODEL_LOCK:
        state.BENCHMARK_PROGRESS = 1
        state.LAST_BENCHMARK_RESULTS = []
        state.MODEL_RUNNING = True

    model_ids = payload.get("ids", [])

    core = ov.Core()
    raw = [d for d in core.available_devices if d not in ["GNA", "HETERO", "MULTI"]]

    pick = {}
    others = []
    for d in raw:
        k = norm_ov_device_name(d)
        if k in ("CPU", "GPU", "NPU"):
            if k not in pick:
                pick[k] = d
        else:
            others.append(d)

    devs = []
    for k in ["CPU", "GPU", "NPU"]:
        if k in pick:
            devs.append(pick[k])
    devs += others

    asyncio.create_task(run_full_matrix_benchmark(model_ids, devs))
    return {"status": "started", "devices": devs}


@router.get("/gui/benchmark/results")
async def get_benchmark_results():
    import app.state as state

    return {
        "results": state.LAST_BENCHMARK_RESULTS,
        "progress": state.BENCHMARK_PROGRESS,
        "running": state.MODEL_RUNNING,
    }


@router.get("/gui/people/list")
async def list_people():
    root = Path(CFG.get("enroll_root", "./enroll"))
    root.mkdir(parents=True, exist_ok=True)
    people_list = []
    for p_dir in root.iterdir():
        if p_dir.is_dir():
            imgs = (
                list(p_dir.glob("*.jpg"))
                + list(p_dir.glob("*.png"))
                + list(p_dir.glob("*.webp"))
            )
            count = len(imgs)
            thumb = imgs[0].name if count > 0 else None
            s_info = STATS.get_info(p_dir.name) or {}
            ts = 0
            if count > 0:
                ts = imgs[0].stat().st_mtime

            people_list.append(
                {
                    "name": p_dir.name,
                    "count": count,
                    "detections": s_info.get("count", 0),
                    "last_seen": s_info.get("last_seen", 0),
                    "thumb": thumb,
                    "sort_ts": ts,
                }
            )
    people_list.sort(key=lambda x: x["sort_ts"], reverse=True)
    return {"people": people_list}


@router.get("/gui/enroll/img/{person}/{file}")
async def get_enroll_img(person: str, file: str):
    try:
        root = Path(CFG.get("enroll_root", "./enroll"))
        p = safe_join(root, person, file)
    except Exception:
        return Response(status_code=400)

    if p.exists():
        return FileResponse(p)
    return Response(status_code=404)


@router.post("/gui/people/add")
async def add_person(payload: PersonPayload):
    name = (payload.name or "").strip()
    if not name:
        return Response(status_code=400)

    try:
        root = Path(CFG.get("enroll_root", "./enroll"))
        p = safe_join(root, name)
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return Response(status_code=400)

    return {"ok": True}


@router.post("/gui/people/delete")
async def del_person(payload: PersonPayload):
    name = (payload.name or "").strip()
    if not name:
        return Response(status_code=400)

    try:
        root = Path(CFG.get("enroll_root", "./enroll"))
        p = safe_join(root, name)

        if p.resolve() == root.resolve():
            return Response(status_code=400)
    except Exception:
        return Response(status_code=400)

    if p.exists():
        shutil.rmtree(p)
    return {"ok": True}


@router.get("/gui/people/files/{name}")
async def list_person_files(name: str):
    nm = (name or "").strip()
    if not nm:
        return {"files": []}

    try:
        root = Path(CFG.get("enroll_root", "./enroll"))
        p = safe_join(root, nm)
    except Exception:
        return {"files": []}

    if not p.exists():
        return {"files": []}

    files = [f.name for f in p.iterdir() if f.is_file()]
    return {"files": files}


@router.post("/gui/people/file/delete")
async def del_person_file(payload: PersonFilePayload):
    person = (payload.person or "").strip()
    file = (payload.file or "").strip()
    if not person or not file:
        return Response(status_code=400)

    try:
        root = Path(CFG.get("enroll_root", "./enroll"))
        p = safe_join(root, person, file)
    except Exception:
        return Response(status_code=400)

    p.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/gui/unknown/list")
async def list_unknown():
    ud = Path(CFG.get("unknown_dir", "./unknown_faces"))
    if not ud.exists():
        return {"files": []}

    image_files = sorted(
        [f for f in ud.iterdir() if f.suffix in [".jpg", ".png"]],
        key=lambda x: x.name,
        reverse=True,
    )[:200]

    result = []
    for f in image_files:
        src = "Unknown"

        json_file = f.with_suffix(".json")
        if json_file.exists():
            try:
                meta = json.loads(json_file.read_text(encoding="utf-8"))
                src = meta.get("source", "Unknown")
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
                        ".\\app\\routers\\gui_api.py:722",
                        "Suppressed exception (was 'pass')",
                    )

        if src == "Unknown":
            parts = f.name.split("_")
            if len(parts) >= 4:
                src = parts[3]

        result.append({"file": f.name, "source": src})

    return {"files": result}


@router.get("/gui/unknown/cluster")
async def cluster_unknown(
    sim_thres: float = 0.75, min_cluster_size: int = 2, limit: int = 2000
):
    ud = Path(CFG.get("unknown_dir", "./unknown_faces"))
    svc = UnknownClusterService(str(ud))
    items, missing = svc.load_items(limit=int(limit))
    clusters = svc.cluster(
        items, sim_thres=float(sim_thres), min_cluster_size=int(min_cluster_size)
    )
    return {
        "clusters": clusters,
        "missing_embeddings": missing,
        "count_items": len(items),
    }


@router.post("/gui/unknown/assign_many")
async def assign_unknown_many(payload: UnknownAssignPayload):
    person = (payload.person or "").strip()
    if not person:
        return Response(status_code=400)

    try:
        ud = Path(CFG.get("unknown_dir", "./unknown_faces"))
        er = Path(CFG.get("enroll_root", "./enroll"))
        person_dir = safe_join(er, person)
        person_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return Response(status_code=400)

    moved = 0

    for f in payload.files or []:
        try:
            src_img = safe_join(ud, f)
            base = src_img.with_suffix("")
            sidecars = [base.with_suffix(".npy"), base.with_suffix(".json")]
            dst_img = safe_join(er, person, Path(f).name)

            if src_img.exists():
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_img), str(dst_img))
                moved += 1

            for sc in sidecars:
                if sc.exists():
                    dst_sc = safe_join(er, person, sc.name)
                    dst_sc.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(sc), str(dst_sc))

        except Exception:
            continue

    return {"ok": True, "moved": moved}


@router.get("/gui/unknown/img/{file}")
async def get_unknown_img(file: str):
    try:
        ud = Path(CFG.get("unknown_dir", "./unknown_faces"))
        p = safe_join(ud, file)
    except Exception:
        return Response(status_code=400)

    if p.exists():
        return FileResponse(p)
    return Response(status_code=404)


@router.post("/gui/unknown/assign")
async def assign_unknown(request: Request):
    pl = await request.json()
    file = (pl.get("file") or "").strip()
    person = (pl.get("person") or "").strip()

    if not file or not person:
        return Response(status_code=400)

    try:
        ud = Path(CFG.get("unknown_dir", "./unknown_faces"))
        er = Path(CFG.get("enroll_root", "./enroll"))

        src = safe_join(ud, file)
        dst = safe_join(er, person, Path(file).name)
    except Exception:
        return Response(status_code=400)

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

        try:
            base = src.with_suffix("")
            for sc in [base.with_suffix(".npy"), base.with_suffix(".json")]:
                if sc.exists():
                    dst_sc = safe_join(er, person, sc.name)
                    dst_sc.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(sc), str(dst_sc))
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
                    ".\\app\\routers\\gui_api.py:833",
                    "Suppressed exception (was 'pass')",
                )

    return {"ok": True}


@router.post("/gui/unknown/clear")
async def clear_unknown():
    ud = Path(CFG.get("unknown_dir", "./unknown_faces"))
    if ud.exists():
        for f in ud.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    return {"ok": True}


@router.post("/gui/enroll/start")
async def start_enroll(request: Request):
    pl = await request.json()
    asyncio.create_task(
        run_enroll_job(
            pl.get("kind"), pl.get("config"), restart_after=(pl.get("kind") == "build")
        )
    )
    return {"ok": True}


@router.get("/gui/enroll/log")
async def get_enroll_log_ep():
    with ENROLL_LOCK:
        return {"lines": list(ENROLL_LOG)}


@router.post("/gui/enroll/clear")
async def clear_enroll_log_ep():
    with ENROLL_LOCK:
        ENROLL_LOG.clear()
    return {"ok": True}


@router.get("/gui/enroll/status")
async def enroll_status_ep():
    return {
        "running": bool(state.ENROLL_RUNNING),
        "lines": len(ENROLL_LOG),
    }


@router.post("/gui/pipeline")
async def toggle_pipe(payload: PipelineTogglePayload):
    with STATE_LOCK:
        state.PIPE_ENABLED = payload.enabled
        CFG["pipeline_enabled"] = state.PIPE_ENABLED
        save_config(CFG)

    if state.PIPE_ENABLED:
        ensure_pipeline_loaded()
    else:
        from app.state import stop_pipeline

        stop_pipeline()
    return {"ok": True}


@router.get("/gui/perf")
async def get_perf():
    return state.PERF


@router.get("/gui/reid/compat")
async def reid_compat(rec_model: str = "", rec_preprocess: str = "auto"):
    import re

    from app.engine.arcface import ArcFaceReID

    with state.STATE_LOCK:
        model_path = rec_model or CFG.get("rec_model") or "./models/w600k_mbf.xml"

    stem = Path(str(model_path)).stem or "model"
    model_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("_") or "model"

    recommended = ArcFaceReID._recommended_for_model(str(model_path))

    root = Path(".")
    faiss_files = sorted(root.glob("*.faiss"))

    compatible: list[str] = []
    dbs: list[dict] = []

    for f in faiss_files:
        meta_path = f.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        md = (meta.get("metadata") or {}) if isinstance(meta, dict) else {}
        db_tag = str(md.get("model_tag") or "").strip()

        dbs.append(
            {
                "file": f.name,
                "meta": meta_path.exists(),
                "model_tag": db_tag,
            }
        )

        if (not db_tag) or (db_tag == model_tag):
            compatible.append(f.name)

    with state.STATE_LOCK:
        cur_db = str(CFG.get("rec_db") or "").strip()
    if cur_db:
        p = Path(cur_db)
        if p.exists():
            name = p.name
            if name not in compatible:
                compatible.append(name)

    return {
        "compatible_db_paths": compatible,
        "recommended_preprocess": recommended,
        "dbs": dbs,
    }


@router.get("/gui/reid/check_compat")
async def check_compat():
    import re

    with state.STATE_LOCK:
        db_path = str(CFG.get("rec_db") or "").strip()
        model_path = str(CFG.get("rec_model") or "./models/w600k_mbf.xml")

    if not db_path:
        return {"status": "warn", "message": "Keine rec_db gesetzt."}

    p = Path(db_path)
    if not p.exists():
        return {"status": "error", "message": f"DB fehlt: {p.name}"}

    stem = Path(model_path).stem or "model"
    model_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("_") or "model"

    meta_path = p.with_suffix(".json")
    if not meta_path.exists():
        return {"status": "ok", "message": "OK (keine Meta-JSON zum Prüfen vorhanden)."}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "warn", "message": "Meta-JSON defekt/unklar."}

    md = (meta.get("metadata") or {}) if isinstance(meta, dict) else {}
    db_tag = str(md.get("model_tag") or "").strip()

    if not db_tag:
        return {"status": "ok", "message": "OK (Meta ohne model_tag)."}

    if db_tag != model_tag:
        return {
            "status": "warn",
            "message": f"DB model_tag='{db_tag}' passt nicht zu Modell '{model_tag}'.",
        }

    return {"status": "ok", "message": "OK"}


@router.get("/gui/facedb/stats")
async def get_facedb_stats():
    from datetime import datetime

    with state.STATE_LOCK:
        db_path_str = CFG.get("rec_db")

    stats = {"exists": False, "count": 0, "size_mb": 0.0, "mtime": "-", "loaded": False}

    if state.PIPE and state.PIPE.facedb:
        stats["count"] = len(state.PIPE.facedb.names)
        stats["loaded"] = True

    if db_path_str:
        p0 = Path(db_path_str)

        if p0.suffix.lower() == ".json":
            idx_path = p0.with_suffix(".faiss")
            meta_path = p0
        elif p0.suffix == "":
            idx_path = p0.with_suffix(".faiss")
            meta_path = p0.with_suffix(".json")
        else:
            idx_path = p0
            meta_path = p0.with_suffix(".json")

        p_stat = (
            idx_path if idx_path.exists() else (meta_path if meta_path.exists() else p0)
        )
        if p_stat.exists():
            stats["exists"] = True
            st = p_stat.stat()
            stats["size_mb"] = round(st.st_size / (1024 * 1024), 2)
            stats["mtime"] = datetime.fromtimestamp(st.st_mtime).strftime(
                "%d.%m.%Y %H:%M:%S"
            )

        if not stats["loaded"] and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    nt = meta.get("ntotal", None)
                    if nt is not None:
                        try:
                            stats["count"] = int(nt)
                        except Exception:
                            pass

                    if not stats["count"]:
                        names = meta.get("names", {}) or {}
                        if isinstance(names, dict):
                            stats["count"] = len(names)
            except Exception:
                pass

    return stats


@router.get("/gui/debug/sanity")
async def gui_debug_sanity(auto_load: bool = True, do_infer: bool = False):
    import numpy as np

    with STATE_LOCK:
        cfg = dict(CFG)
        enabled = bool(getattr(state, "PIPE_ENABLED", False))
        pipe = getattr(state, "PIPE", None)

    if auto_load and enabled and pipe is None:
        try:
            ensure_pipeline_loaded()
        except Exception as e:
            return {
                "ok": False,
                "stage": "ensure_pipeline_loaded",
                "error": str(e),
                "pipeline_enabled": enabled,
            }

        with STATE_LOCK:
            pipe = getattr(state, "PIPE", None)

    core_devices = []
    try:
        if GLOBAL_CORE:
            core_devices = list(GLOBAL_CORE.available_devices)
    except Exception:
        pass

    report = {
        "ok": True,
        "pipeline_enabled": enabled,
        "pipeline_loaded": bool(pipe is not None),
        "devices": core_devices,
        "cfg": {
            "device": cfg.get("device"),
            "device_yolo": cfg.get("device_yolo"),
            "device_face_det": cfg.get("device_face_det"),
            "device_reid": cfg.get("device_reid"),
            "yolo_model": cfg.get("yolo_model"),
            "face_det_model": cfg.get("face_det_model"),
            "rec_model": cfg.get("rec_model"),
            "rec_db": cfg.get("rec_db"),
        },
        "runtime": {"models": []},
    }

    if pipe is not None:

        def _add(role: str, ov):
            if ov is None:
                report["runtime"]["models"].append({"role": role, "loaded": False})
                return
            try:
                info = ov.get_pool_info() if hasattr(ov, "get_pool_info") else {}
                info["role"] = role
                info["loaded"] = True
                report["runtime"]["models"].append(info)
            except Exception as e:
                report["runtime"]["models"].append(
                    {"role": role, "loaded": True, "error": str(e)}
                )

        _add("YOLO", getattr(pipe, "yolo_ov", None))
        fd = getattr(pipe, "face_det", None)
        _add("FaceDet", getattr(fd, "ov", None) if fd else None)
        reid = getattr(pipe, "reid", None)
        _add("ReID", getattr(reid, "ov", None) if reid else None)

        try:
            db = getattr(pipe, "facedb", None)
            report["runtime"]["facedb"] = {
                "loaded": bool(db is not None),
                "count": int(len(getattr(db, "names", {}) or {})) if db else 0,
                "dim": getattr(db, "dim", None) if db else None,
            }
        except Exception:
            pass

    if do_infer:
        if pipe is None:
            report["ok"] = False
            report["test"] = {"ok": False, "error": "pipeline not loaded"}
            return report

        def _run_tests():
            out = {"ok": True, "tests": {}}
            img = np.zeros((640, 640, 3), dtype=np.uint8)

            try:
                dets, dbg = pipe.yolo.detect(img)
                out["tests"]["yolo"] = {"ok": True, "dets": int(len(dets)), "dbg": dbg}
            except Exception as e:
                out["ok"] = False
                out["tests"]["yolo"] = {"ok": False, "error": str(e)}

            try:
                if pipe.face_det is None:
                    out["tests"]["face_det"] = {
                        "ok": True,
                        "skipped": True,
                        "reason": "face_det disabled/not loaded",
                    }
                else:
                    dets, dbg = pipe.face_det.detect(img)
                    out["tests"]["face_det"] = {
                        "ok": True,
                        "dets": int(len(dets)),
                        "dbg": dbg,
                    }
            except Exception as e:
                out["ok"] = False
                out["tests"]["face_det"] = {"ok": False, "error": str(e)}

            try:
                if pipe.reid is None:
                    out["tests"]["reid"] = {
                        "ok": True,
                        "skipped": True,
                        "reason": "reid disabled/not loaded",
                    }
                else:
                    emb = pipe.reid.embed(np.zeros((112, 112, 3), dtype=np.uint8))
                    out["tests"]["reid"] = {
                        "ok": True,
                        "emb_dim": int(getattr(emb, "shape", [0])[0]),
                    }
            except Exception as e:
                out["ok"] = False
                out["tests"]["reid"] = {"ok": False, "error": str(e)}

            return out

        test = await asyncio.to_thread(_run_tests)
        report["test"] = test
        report["ok"] = bool(report["ok"] and test.get("ok", False))

    return report


@router.post("/gui/debug/sanity_image")
async def gui_debug_sanity_image(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Could not decode image"}

    with STATE_LOCK:
        enabled = bool(getattr(state, "PIPE_ENABLED", False))
        pipe = getattr(state, "PIPE", None)

    if enabled and pipe is None:
        try:
            ensure_pipeline_loaded()
        except Exception as e:
            return {"ok": False, "stage": "ensure_pipeline_loaded", "error": str(e)}

        with STATE_LOCK:
            pipe = getattr(state, "PIPE", None)

    if pipe is None:
        return {"ok": False, "error": "pipeline not loaded"}

    out = {"ok": True, "img_shape": list(img.shape), "tests": {}}

    try:
        dets, dbg = pipe.yolo.detect(img)
        out["tests"]["yolo"] = {"ok": True, "dets": int(len(dets)), "dbg": dbg}
    except Exception as e:
        out["ok"] = False
        out["tests"]["yolo"] = {"ok": False, "error": str(e)}

    try:
        if pipe.face_det is None:
            out["tests"]["face_det"] = {
                "ok": True,
                "skipped": True,
                "reason": "face_det disabled/not loaded",
            }
        else:
            dets, dbg = pipe.face_det.detect(img)
            out["tests"]["face_det"] = {"ok": True, "dets": int(len(dets)), "dbg": dbg}
    except Exception as e:
        out["ok"] = False
        out["tests"]["face_det"] = {"ok": False, "error": str(e)}

    return out
