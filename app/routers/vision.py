import asyncio
import logging
import shutil
import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

import app.state as state
from app.schemas import Prediction, VisionResponse
from app.services.facedb import run_enroll_job
from app.state import CFG
from app.utils import decode_image_bytes, draw_predictions_on_image

API_LOG = logging.getLogger("obf.api_spy")

router = APIRouter()

_BI_DEBUG_RING = deque(maxlen=50)


@router.post("/v1/vision/custom/list")
@router.get("/v1/vision/custom/list")
async def bi_custom_list(request: Request):
    try:
        models = CFG.get("bi_custom_models")

        if isinstance(models, str):
            raw = models.replace(";", ",").replace("\n", ",")
            models = [m.strip() for m in raw.split(",") if m.strip()]

        if not isinstance(models, list) or not models:
            models = ["Einfahrt", "Garten", "Haustuer"]

        seen = set()
        clean = []
        for m in models:
            s = str(m).strip()
            if not s:
                continue
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            clean.append(s)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "models": clean,
                "moduleId": "OBF",
                "moduleName": "OBF Vision",
                "command": "custom/list",
                "code": 200,
                "message": "OK",
            },
        )

    except Exception as e:
        API_LOG.exception("bi_custom_list failed")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "models": [],
                "moduleId": "OBF",
                "command": "custom/list",
                "code": 500,
                "message": str(e),
            },
        )


def _sanitize_source(name: str) -> str:
    import re

    s = (name or "").strip()
    if not s:
        return "BI_Unknown"

    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = s.strip("._-")

    return s if s else "BI_Unknown"


def _bi_debug_push(entry: dict) -> None:
    try:
        _BI_DEBUG_RING.appendleft(entry)
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
                ".\\app\\routers\\vision.py:112",
                "Suppressed exception (was 'pass')",
            )


def _summarize_form(form) -> dict:
    out = {"fields": {}, "files": {}}
    try:
        items = (
            list(form.multi_items())
            if hasattr(form, "multi_items")
            else list(form.items())
        )
    except Exception:
        items = []

    for k, v in items:
        try:
            fn = getattr(v, "filename", None)
            ct = getattr(v, "content_type", None)
            if fn is not None:
                out["files"].setdefault(k, []).append(
                    {
                        "filename": str(fn),
                        "content_type": str(ct) if ct is not None else None,
                    }
                )
            else:
                s = str(v)
                if len(s) > 300:
                    s = s[:300] + "…"
                out["fields"].setdefault(k, []).append(s)
        except Exception:
            continue
    return out


@router.get("/gui/bi/debug")
async def gui_bi_debug():
    return {
        "enabled": bool(CFG.get("debug_bi_requests", False)),
        "count": len(_BI_DEBUG_RING),
        "items": list(_BI_DEBUG_RING),
    }


def build_vision_response(
    ok, preds, msg, code, cmd, t_inf, t_proc, module="OBF Vision"
):
    pydantic_preds = []
    for p in preds:
        pydantic_preds.append(
            Prediction(
                label=str(p["label"]),
                confidence=float(p["confidence"]),
                x_min=int(p["x_min"]),
                y_min=int(p["y_min"]),
                x_max=int(p["x_max"]),
                y_max=int(p["y_max"]),
                userid=p.get("userid"),
            )
        )

    return VisionResponse(
        success=ok,
        message=msg,
        count=len(pydantic_preds),
        predictions=pydantic_preds,
        inferenceMs=int(t_inf),
        processMs=int(t_proc),
        analysisRoundTripMs=int(t_proc),
        moduleName=module,
        code=code,
        command=cmd,
        requestId=str(uuid.uuid4()),
        timestampUTC=str(time.time()),
    )


async def extract_image(request: Request, image: UploadFile):
    debug_enabled = bool(CFG.get("debug_bi_requests", False))

    dbg = {
        "ts": time.time(),
        "method": request.method,
        "path": str(request.url.path),
        "client": getattr(getattr(request, "client", None), "host", None),
        "content_type": request.headers.get("content-type"),
        "content_length": request.headers.get("content-length"),
        "query": dict(request.query_params),
        "headers": {
            "user-agent": request.headers.get("user-agent"),
            "x-forwarded-for": request.headers.get("x-forwarded-for"),
            "x-real-ip": request.headers.get("x-real-ip"),
        },
        "multipart": None,
        "picked_image_field": None,
        "picked_image_filename": None,
        "min_conf_candidates": {},
        "image_bytes_len": 0,
    }

    ct = (request.headers.get("content-type") or "").lower()

    form = None
    if "multipart/form-data" in ct:
        try:
            form = await request.form()
            dbg["multipart"] = _summarize_form(form)

            for key in (
                "min_confidence",
                "minConfidence",
                "confidence",
                "min-confidence",
                "MinConfidence",
                "camera",
                "cam",
                "source",
                "clip",
                "profile",
                "alert",
            ):
                if key in form:
                    try:
                        dbg["min_conf_candidates"][key] = str(form.get(key))
                    except Exception:
                        dbg["min_conf_candidates"][key] = "<unreadable>"
        except Exception as e:
            dbg["multipart"] = {"error": str(e)}
            form = None

    up = None
    if image is not None:
        up = image
        dbg["picked_image_field"] = "image(param)"
        dbg["picked_image_filename"] = getattr(image, "filename", None)
    elif form is not None:
        for k in ("image", "file", "snapshot", "img", "data", "upload"):
            v = form.get(k)
            if getattr(v, "filename", None) is not None:
                up = v
                dbg["picked_image_field"] = k
                dbg["picked_image_filename"] = getattr(v, "filename", None)
                break

        if up is None:
            try:
                items = (
                    list(form.multi_items())
                    if hasattr(form, "multi_items")
                    else list(form.items())
                )
                for k, v in items:
                    if getattr(v, "filename", None) is not None:
                        up = v
                        dbg["picked_image_field"] = k
                        dbg["picked_image_filename"] = getattr(v, "filename", None)
                        break
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
                        ".\\app\\routers\\vision.py:281",
                        "Suppressed exception (was 'pass')",
                    )

    if up is not None:
        b = await up.read()
        dbg["image_bytes_len"] = len(b or b"")
        if debug_enabled:
            API_LOG.info("[BI-DBG] %s", dbg)
            _bi_debug_push(dbg)
        return b

    b = await request.body()
    dbg["picked_image_field"] = "raw-body"
    dbg["image_bytes_len"] = len(b or b"")
    if debug_enabled:
        API_LOG.info("[BI-DBG] %s", dbg)
        _bi_debug_push(dbg)
    return b


@router.post("/v1/vision/detection", response_model=VisionResponse)
@router.post("/v1/vision/custom/{model_name}", response_model=VisionResponse)
async def vision_handler(
    request: Request,
    model_name: str = "detection",
    image: UploadFile = File(None),
    min_confidence: str | None = Form(None),
):
    t0 = time.perf_counter()

    if not state.PIPE_ENABLED or state.PIPE is None:
        with state.STATE_LOCK:
            state.PERF.update(False, 0.0, [])
        return build_vision_response(False, [], "Pipeline stopped", 503, "det", 0, 0)

    try:
        img_bytes = await extract_image(request, image)
        img_bgr = await run_in_threadpool(decode_image_bytes, img_bytes)
    except Exception as e:
        with state.STATE_LOCK:
            state.PERF.update(False, 0.0, [])
        return build_vision_response(False, [], f"Bad Image: {e}", 400, "det", 0, 0)

    pipe = state.PIPE

    path = str(request.url.path or "")
    is_custom = path.startswith("/v1/vision/custom/")

    if is_custom:
        detected_source = _sanitize_source(model_name)
    else:
        detected_source = "BI_detection"

        if min_confidence is None:
            try:
                form = await request.form()
                for key in (
                    "min_confidence",
                    "minConfidence",
                    "confidence",
                    "min-confidence",
                    "MinConfidence",
                ):
                    if key in form:
                        min_confidence = str(form.get(key))
                        break
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
                        ".\\app\\routers\\vision.py:345",
                        "Suppressed exception (was 'pass')",
                    )

    snapshot_mode = True
    conf_thres = 0.40
    if min_confidence is not None:
        try:
            v = float(str(min_confidence).replace(",", "."))
            if v > 0:
                conf_thres = v
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
                    ".\\app\\routers\\vision.py:358",
                    "Suppressed exception (was 'pass')",
                )

    try:
        preds_raw, dbg = await asyncio.to_thread(
            pipe.run, img_bgr, detected_source, snapshot_mode=snapshot_mode
        )

        preds = []
        labels = []
        for p in preds_raw or []:
            lbl = str(p.get("label") or "")
            lbl_norm = lbl.strip().lower()

            if lbl_norm in ("blurry", "poorquality"):
                continue

            if lbl_norm in ("", "face"):
                p["label"] = "Unknown"
                p["userid"] = None
                lbl = "Unknown"

            if float(p.get("confidence", 0.0) or 0.0) < float(conf_thres):
                continue

            labels.append(lbl)
            preds.append(p)

        t_proc = (time.perf_counter() - t0) * 1000.0
        t_inf = 0.0
        try:
            t_inf = float((dbg or {}).get("yolo", {}).get("time_ms", 0.0))
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
                    ".\\app\\routers\\vision.py:381",
                    "Suppressed exception (was 'pass')",
                )

        with state.STATE_LOCK:
            state.PERF.update(True, t_proc, labels)

        return build_vision_response(
            True, preds, f"OK ({detected_source})", 200, model_name, t_inf, t_proc
        )

    except Exception as e:
        t_proc = (time.perf_counter() - t0) * 1000.0

        with state.STATE_LOCK:
            state.PERF.update(False, t_proc, [])

        return build_vision_response(
            False, [], f"Pipeline error: {e}", 500, model_name, 0, t_proc
        )


@router.post("/v1/vision/face/recognize")
@router.post("/v1/vision/face")
async def face_recognize(
    request: Request,
    image: UploadFile = File(None),
    min_confidence: float | None = Form(0.40),
):
    t0 = time.perf_counter()
    if not state.PIPE_ENABLED or state.PIPE is None:
        return JSONResponse(
            {"success": False, "message": "Pipeline stopped"}, status_code=503
        )

    try:
        img_bytes = await extract_image(request, image)
        img_bgr = await run_in_threadpool(decode_image_bytes, img_bytes)
    except Exception as e:
        return JSONResponse(
            {"success": False, "message": f"Bad Image: {e}"}, status_code=400
        )

    pipe = state.PIPE
    try:
        preds_raw, dbg = await asyncio.to_thread(pipe.run, img_bgr, "API_Face")
        face_preds = []
        found_names = []

        for p in preds_raw:
            lbl = p["label"]
            if lbl in ["Blurry", "Unknown", "face"]:
                continue

            conf = float(p["confidence"])

            if conf >= min_confidence:
                face_preds.append(
                    {
                        "confidence": conf,
                        "userid": lbl,
                        "label": lbl,
                        "x_min": int(p["x_min"]),
                        "y_min": int(p["y_min"]),
                        "x_max": int(p["x_max"]),
                        "y_max": int(p["y_max"]),
                    }
                )

                found_names.append(lbl)

        dt = int((time.perf_counter() - t0) * 1000)

        return {
            "success": True,
            "predictions": face_preds,
            "message": "Recognized: "
            + (", ".join(set(found_names)) if found_names else "None"),
            "code": 200,
            "inferenceMs": dbg.get("faces", {}).get("time_ms", 0),
            "processMs": dt,
            "analysisRoundTripMs": dt,
            "moduleId": "FaceProcessing",
            "moduleName": "Face Processing",
        }
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@router.post("/v1/vision/face/register")
async def face_register(userid: str = Form(...), image: list[UploadFile] = File(...)):
    API_LOG.warning(f"⚡ REGISTER REQUEST empfangen! User: '{userid}'")
    if not userid:
        return JSONResponse(
            {"success": False, "error": "userid missing"}, status_code=400
        )
    root = Path(CFG.get("enroll_root", "./enroll"))
    person_dir = root / userid
    person_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    import cv2

    for img_file in image:
        try:
            content = await img_file.read()
            fname = f"{uuid.uuid4().hex}.jpg"
            img_bgr = await run_in_threadpool(decode_image_bytes, content)
            target_path = person_dir / fname
            ok = cv2.imwrite(str(target_path), img_bgr)
            if not ok:
                target_path.write_bytes(content)
            saved_count += 1
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
                    ".\\app\\routers\\vision.py:546",
                    "Suppressed exception (was 'pass')",
                )
    if saved_count > 0:
        API_LOG.info(f" -> {saved_count} Bilder gespeichert. Trigger Build DB...")

        asyncio.create_task(run_enroll_job("build", {}, restart_after=False))
        return JSONResponse(
            {"success": True, "message": "face added", "count": saved_count}
        )
    return JSONResponse(
        {"success": False, "error": "No valid images received"}, status_code=400
    )


@router.post("/v1/vision/face/list")
async def face_list():
    return JSONResponse({"success": True, "faces": []})


@router.post("/v1/vision/face/delete")
async def face_delete(request: Request):
    form = await request.form()
    userid = form.get("userid")
    if not userid:
        return JSONResponse({"success": False, "error": "userid missing"})
    root = Path(CFG.get("enroll_root", "./enroll"))
    person_dir = root / userid
    if person_dir.exists() and person_dir.is_dir():
        try:
            shutil.rmtree(person_dir)

            asyncio.create_task(run_enroll_job("build", {}, restart_after=False))
            return JSONResponse({"success": True, "message": "deleted"})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)})
    return JSONResponse({"success": False, "error": "User not found"})


def generate_mjpeg_stream(rtsp_url: str, min_conf: float):
    import os
    import time

    import cv2

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    frame_count = 0
    fail_count = 0
    process_every_n = 3

    if not cap.isOpened():
        yield (b"--frame\r\nContent-Type: text/plain\r\n\r\nConnection failed\r\n")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                fail_count += 1
                if fail_count > 50:
                    print(f"Stream aborted: {rtsp_url} (Too many read errors)")
                    break

                time.sleep(0.1)
                continue

            fail_count = 0

            frame_count += 1

            if state.PIPE and (frame_count % process_every_n == 0):
                try:
                    t0 = time.perf_counter()
                    preds_raw, _dbg = state.PIPE.run(frame, "StreamTest")
                    dt_ms = (time.perf_counter() - t0) * 1000.0

                    labels = []
                    for p in preds_raw or []:
                        lab = p.get("label")
                        if lab and lab not in labels:
                            labels.append(str(lab))

                    state.PERF.update(True, dt_ms, labels)

                    preds_filtered = [
                        p
                        for p in (preds_raw or [])
                        if float(p.get("confidence", 0.0)) >= float(min_conf)
                    ]
                    frame = draw_predictions_on_image(frame, preds_filtered)

                except Exception:
                    try:
                        state.PERF.update(False, 0.0, [])
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
                                ".\\app\\routers\\vision.py:626",
                                "Suppressed exception (was 'pass')",
                            )

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

            time.sleep(0.05)

    finally:
        cap.release()


TEMP_VIDEO_DIR = Path("temp_videos")
TEMP_VIDEO_DIR.mkdir(exist_ok=True)


def cleanup_video(path: Path) -> None:
    try:
        if path and path.exists():
            path.unlink(missing_ok=True)
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
                ".\\app\\routers\\vision.py:651",
                "Suppressed exception (was 'pass')",
            )


def _safe_suffix(filename: str) -> str:
    try:
        suf = (Path(filename).suffix or "").lower().strip()
        if not suf or len(suf) > 8:
            return ".mp4"

        if not all(c.isalnum() or c == "." for c in suf):
            return ".mp4"
        if not suf.startswith("."):
            suf = "." + suf
        return suf
    except Exception:
        return ".mp4"


def _find_video_path(token: str) -> Path | None:
    try:
        if not token:
            return None
        for p in TEMP_VIDEO_DIR.glob(f"{token}.*"):
            if p.is_file():
                return p
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
                ".\\app\\routers\\vision.py:684",
                "Suppressed exception (was 'pass')",
            )
    return None


@router.post("/v1/vision/video/upload")
async def upload_video_test(file: UploadFile = File(...)):
    from fastapi import HTTPException

    MAX_VIDEO_SIZE = 500 * 1024 * 1024
    CHUNK_SIZE = 64 * 1024

    try:
        token = uuid.uuid4().hex
        suf = _safe_suffix(getattr(file, "filename", "") or "")
        file_path = TEMP_VIDEO_DIR / f"{token}{suf}"

        total_size = 0

        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_VIDEO_SIZE:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video zu groß (max {MAX_VIDEO_SIZE // (1024 * 1024)} MB)",
                    )

                buffer.write(chunk)

        LOG.info(f"Video uploaded: {token} ({total_size / (1024 * 1024):.1f} MB)")
        return {"success": True, "token": token}

    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("Video upload failed")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


def generate_video_stream(video_path: Path, min_conf: float):
    import time

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    process_every_n = 2
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            if state.PIPE and (frame_count % process_every_n == 0):
                try:
                    t0 = time.perf_counter()
                    preds_raw, _ = state.PIPE.run(frame, "VideoFile")
                    dt_ms = (time.perf_counter() - t0) * 1000.0

                    try:
                        labels = []
                        for p in preds_raw or []:
                            lab = p.get("label")
                            if lab and lab not in labels:
                                labels.append(str(lab))
                        state.PERF.update(True, dt_ms, labels)
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
                                ".\\app\\routers\\vision.py:737",
                                "Suppressed exception (was 'pass')",
                            )

                    preds_filtered = [
                        p
                        for p in (preds_raw or [])
                        if float(p.get("confidence", 0.0)) >= float(min_conf)
                    ]
                    frame = draw_predictions_on_image(frame, preds_filtered)
                except Exception:
                    try:
                        state.PERF.update(False, 0.0, [])
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
                                ".\\app\\routers\\vision.py:749",
                                "Suppressed exception (was 'pass')",
                            )

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            time.sleep(0.01)

    finally:
        try:
            cap.release()
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
                    ".\\app\\routers\\vision.py:764",
                    "Suppressed exception (was 'pass')",
                )


@router.get("/v1/vision/video/stream")
async def stream_video_file(
    token: str, min_conf: float = 0.40, background_tasks: BackgroundTasks = None
):
    video_path = _find_video_path(token)
    if not video_path:
        return JSONResponse(
            {"success": False, "error": "Video token not found"}, status_code=404
        )

    if background_tasks is not None:
        background_tasks.add_task(cleanup_video, video_path)

    return StreamingResponse(
        generate_video_stream(video_path, float(min_conf)),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


def _is_private_ip(ip: str) -> bool:
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)

        if addr.is_loopback:
            return True
        return addr.is_private or addr.is_link_local
    except Exception:
        return False


def _validate_rtsp_url(url: str) -> str:
    from urllib.parse import urlsplit

    global CFG

    u = (url or "").strip()
    if not u or len(u) > 2048:
        raise ValueError("Invalid URL")

    parts = urlsplit(u)
    if parts.scheme not in ("rtsp", "rtsps"):
        raise ValueError("Only rtsp:// or rtsps:// allowed")

    host = parts.hostname
    if not host:
        raise ValueError("Missing host")

    allow_public = bool(CFG.get("stream_test_allow_public", False))

    if not allow_public and _is_private_ip(host) is False:
        allowed_hosts = set(CFG.get("stream_test_allowed_hosts", []) or [])
        if host not in allowed_hosts and not host.endswith(".local"):
            raise ValueError("RTSP host blocked (not private / not allowed)")

    return u


@router.get("/v1/vision/stream_test")
async def stream_rtsp_test(url: str, min_conf: float = 0.40):
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse

    try:
        safe_url = _validate_rtsp_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return StreamingResponse(
        generate_mjpeg_stream(safe_url, min_conf),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )
