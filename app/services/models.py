import asyncio
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov

import app.state as state
from app.state import MODEL_LOCK, log_model

_ASYNC_MODEL_LOCK = asyncio.Lock()


MODEL_CATALOG = {
    "yolo11n": {
        "type": "yolo",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "desc": "YOLOv11 Nano. Schnellste Inferenz.",
        "file": "yolo11n.xml",
    },
    "yolo11s": {
        "type": "yolo",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "desc": "YOLOv11 Small. Standard-Wahl.",
        "file": "yolo11s.xml",
    },
    "yolo11m": {
        "type": "yolo",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "desc": "YOLOv11 Medium. Hohe Genauigkeit.",
        "file": "yolo11m.xml",
    },
    "yolov8n": {
        "type": "yolo",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "desc": "YOLOv8 Nano (Legacy).",
        "file": "yolov8n.xml",
    },
    "yolov8s": {
        "type": "yolo",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        "desc": "YOLOv8 Small (Legacy).",
        "file": "yolov8s.xml",
    },
    "yolov8n-face": {
        "type": "face_det",
        "url": "https://drive.google.com/uc?export=download&id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
        "desc": "Gesichtserkennung (PT -> OpenVINO via Ultralytics).",
        "file": "yolov8n-face.xml",
        "is_pt": True,
    },
    "w600k_mbf": {
        "type": "reid",
        "url": "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx",
        "desc": "Face Re-ID (MobileFaceNet).",
        "file": "w600k_mbf.xml",
        "is_onnx": True,
    },
    "retail-0095": {
        "type": "reid",
        "omz_name": "face-reidentification-retail-0095",
        "desc": "Intel Retail ReID.",
        "file": "face-reidentification-retail-0095.xml",
    },
    "resnet100": {
        "type": "reid",
        "omz_name": "face-recognition-resnet100-arcface-onnx",
        "desc": "ArcFace ResNet100 (Präzise).",
        "file": "face-recognition-resnet100-arcface-onnx.xml",
    },
}


def get_models_dir() -> Path:
    p = Path("./models")
    p.mkdir(parents=True, exist_ok=True)
    return p


def delete_model_files(model_id: str) -> bool:
    info = MODEL_CATALOG.get(model_id)
    if not info:
        return False

    models_dir = get_models_dir()
    base = Path(info["file"]).stem
    deleted = False

    for f in models_dir.glob(f"{base}*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
            deleted = True
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
                    ".\\app\\services\\models.py:50",
                    "Suppressed exception (was 'pass')",
                )

    return deleted


def get_models_storage_info():
    m_dir = get_models_dir()
    files = [f for f in m_dir.glob("**/*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    return {"size_mb": round(total / (1024 * 1024), 2), "file_count": len(files)}


def _download_file(url: str, dest: Path) -> None:
    log_model(f"Download: {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
        log_model("Download fertig.")
    except Exception as e:
        log_model(f"Download Fehler: {e}")
        raise


def _run_cmd_stream(cmd: list[str], cwd: Path | None = None) -> int:
    log_model(f"CMD: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    for line in proc.stdout:
        log_model(line.rstrip())

    return proc.wait()


def yolo_convert_to_openvino(pt_file: str):
    out_dir = get_models_dir()
    pt_path = Path(pt_file).resolve()

    rc = _run_cmd_stream(
        [
            sys.executable,
            "-c",
            f"from ultralytics import YOLO; m=YOLO(r'{str(pt_path)}'); m.export(format='openvino', dynamic=False, half=True)",
        ],
        cwd=out_dir,
    )

    ov_dir = pt_path.parent / f"{pt_path.stem}_openvino_model"
    if ov_dir.exists() and ov_dir.is_dir():
        log_model(f"Verschiebe Dateien aus {ov_dir.name}...")
        for f in ov_dir.glob("*"):
            shutil.move(str(f), str(out_dir / f.name))
        shutil.rmtree(ov_dir)

        xml_cand = out_dir / f"{pt_path.stem}.xml"
        if xml_cand.exists():
            log_model("Konvertierung erfolgreich.")
            return

    if rc != 0:
        raise RuntimeError("Ultralytics export failed.")


def omz_download_convert_to_models(omz_name: str, precision: str = "FP16"):
    cache = Path("./_omz_cache")

    if shutil.which("omz_downloader") is None or shutil.which("omz_converter") is None:
        msg = (
            "OMZ Tools fehlen: 'omz_downloader'/'omz_converter' nicht gefunden.\n"
            "Installiere die Open Model Zoo Tools (z.B. über openvino-dev / omz-tools) "
            "oder stelle sicher, dass sie im PATH sind."
        )
        log_model(msg)
        raise RuntimeError(msg)

    _run_cmd_stream(
        [
            "omz_downloader",
            "--name",
            omz_name,
            "--precision",
            precision,
            "--output_dir",
            str(cache),
        ]
    )
    _run_cmd_stream(
        [
            "omz_converter",
            "--name",
            omz_name,
            "--precisions",
            precision,
            "--download_dir",
            str(cache),
            "--output_dir",
            str(cache),
        ]
    )

    xml_files = list(cache.rglob(f"{omz_name}.xml"))
    if xml_files:
        xml = xml_files[0]
        tgt_xml = get_models_dir() / xml.name
        tgt_bin = get_models_dir() / xml.with_suffix(".bin").name

        shutil.copy2(xml, tgt_xml)
        shutil.copy2(xml.with_suffix(".bin"), tgt_bin)
        log_model(f"Installiert: {tgt_xml.name}")
    else:
        log_model("Fehler: OMZ Konvertierung lieferte keine XML.")
        raise RuntimeError("OMZ conversion produced no XML output.")


@contextmanager
def _silence_torch_pt2_probe_for_onnx(model_path: str):
    try:
        import os

        _, ext = os.path.splitext(model_path)
        if ext.lower() != ".onnx":
            yield
            return

        import torch

        if not (hasattr(torch, "export") and hasattr(torch.export, "load")):
            yield
            return

        orig = torch.export.load

        def _skip_pt2(*args, **kwargs):
            raise RuntimeError("skip pt2 probe for onnx")

        torch.export.load = _skip_pt2
        yield
    finally:
        try:
            import torch

            torch.export.load = orig
        except Exception:
            pass


def _process_yolo_job(info, mid, out_dir):
    pt_dest = out_dir / (mid + ".pt")
    if not pt_dest.exists():
        _download_file(info["url"], pt_dest)
    yolo_convert_to_openvino(str(pt_dest))


def _process_onnx_job(info, mid, out_dir):
    onnx_dest = out_dir / (mid + ".onnx")
    if not onnx_dest.exists():
        _download_file(info["url"], onnx_dest)

    log_model("Konvertiere ONNX zu OpenVINO IR...")

    with _silence_torch_pt2_probe_for_onnx(str(onnx_dest)):
        model = ov.convert_model(str(onnx_dest))
    ov.save_model(model, str(out_dir / info["file"]), compress_to_fp16=True)
    log_model("Fertig.")


def _process_pt_job(info, mid, out_dir):
    base = Path(info["file"]).stem
    pt_dest = out_dir / f"{base}.pt"

    if not pt_dest.exists():
        _download_file(info["url"], pt_dest)

    yolo_convert_to_openvino(str(pt_dest))


async def run_model_job(kind: str, payload: dict[str, Any]) -> None:
    async with _ASYNC_MODEL_LOCK:
        with MODEL_LOCK:
            if state.MODEL_RUNNING:
                log_model("Job läuft bereits, überspringe Request.")
                return
            state.MODEL_RUNNING = True

        try:
            selection = payload.get("selection", [])
            out_dir = get_models_dir()

            for mid in selection:
                info = MODEL_CATALOG.get(mid)
                if not info:
                    continue

                xml = out_dir / info["file"]
                binf = out_dir / (Path(info["file"]).with_suffix(".bin").name)

                if (
                    xml.exists()
                    and binf.exists()
                    and xml.stat().st_size > 0
                    and binf.stat().st_size > 0
                ):
                    log_model(f"{mid} ist bereits installiert.")
                    continue

                if (out_dir / info["file"]).exists():
                    log_model(f"{mid} ist bereits installiert.")
                    continue

                log_model(f"\n=== Setup: {info['desc']} ===")

                if "omz_name" in info:
                    await asyncio.to_thread(
                        omz_download_convert_to_models, info["omz_name"]
                    )

                elif info.get("is_pt"):
                    log_model("Prüfe pip Package 'ultralytics'.")
                    await asyncio.to_thread(
                        _run_cmd_stream,
                        [sys.executable, "-m", "pip", "install", "-U", "ultralytics"],
                    )
                    await asyncio.to_thread(_process_pt_job, info, mid, out_dir)

                elif info.get("is_onnx"):
                    await asyncio.to_thread(_process_onnx_job, info, mid, out_dir)

                elif info.get("type") == "yolo":
                    log_model("Prüfe pip Package 'ultralytics'...")
                    await asyncio.to_thread(
                        _run_cmd_stream,
                        [sys.executable, "-m", "pip", "install", "-U", "ultralytics"],
                    )
                    await asyncio.to_thread(_process_yolo_job, info, mid, out_dir)

            log_model("\nSetup beendet. Download finished - Please Restart")

        except Exception as e:
            log_model(f"CRITICAL ERROR: {e}")
            import traceback

            log_model(traceback.format_exc())
        finally:
            with MODEL_LOCK:
                state.MODEL_RUNNING = False


async def run_full_matrix_benchmark(model_ids: list, devices: list):
    from openvino import properties

    import app.state as state

    def _norm_dev_name(d: str) -> str:
        s = str(d or "").strip()
        if not s:
            return s
        base = s.split(":")[0].split(".")[0].strip()
        up = base.upper()
        if up in ("CPU", "GPU", "NPU"):
            return up
        return base

    def _np_dtype_from_ov_et(et) -> np.dtype:
        s = str(et).lower()
        if "u8" in s:
            return np.uint8
        if "i8" in s:
            return np.int8
        if "f16" in s:
            return np.float16
        if "f32" in s:
            return np.float32
        return np.float32

    def _make_perf_config(mode: str) -> dict:
        mode = (mode or "LATENCY").upper().strip()
        cfg = {}
        try:
            pm = (
                properties.hint.PerformanceMode.THROUGHPUT
                if mode == "THROUGHPUT"
                else properties.hint.PerformanceMode.LATENCY
            )
            cfg[properties.hint.performance_mode()] = pm
        except Exception:
            cfg["PERFORMANCE_HINT"] = mode
        return cfg

    def _get_optimal_nireq(compiled) -> int:
        try:
            v = int(compiled.get_property(properties.optimal_number_of_infer_requests))
            if v > 0:
                return v
        except Exception:
            pass
        try:
            v = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
            if v > 0:
                return v
        except Exception:
            pass
        return 1

    def _cfg_set(cfg: dict, key_obj_getter, key_str: str, value):
        try:
            k = key_obj_getter()
            cfg[k] = value
        except Exception:
            cfg[key_str] = value

    lat_budget_s = float(os.getenv("OBF_BENCH_LAT_SEC", "1.0"))
    thr_budget_s = float(os.getenv("OBF_BENCH_THR_SEC", "1.0"))
    max_iters = int(os.getenv("OBF_BENCH_MAX_ITERS", "30"))
    max_nireq_cap = int(os.getenv("OBF_BENCH_MAX_IREQ", "16"))
    gpu_thr_cap = int(os.getenv("OBF_BENCH_GPU_THR_MAX_IREQ", "4"))

    try:
        core = ov.Core()
        log_model("=== Start: Benchmark (LATENCY + THROUGHPUT) ===")
    except Exception:
        state.BENCHMARK_PROGRESS = 100
        return []

    total_steps = max(1, len(model_ids) * max(1, len(devices)) * 2)
    current_step = 0

    matrix_results = []
    md = get_models_dir()

    for mid in model_ids:
        info = MODEL_CATALOG.get(mid)
        if not info:
            continue
        xml_path = md / info["file"]
        if not xml_path.exists():
            continue

        model_row = {"id": mid, "benchmarks": {}}

        try:
            model = core.read_model(model=str(xml_path))

            input_layer = model.input(0)
            p_shape = input_layer.get_partial_shape()

            shape = []
            for dim in p_shape:
                if dim.is_dynamic:
                    if len(shape) == 0:
                        shape.append(1)
                    elif len(shape) == 1:
                        shape.append(3)
                    else:
                        shape.append(640)
                else:
                    shape.append(dim.get_length())

            dtype = _np_dtype_from_ov_et(input_layer.get_element_type())

            if dtype in (np.uint8,):
                test_input = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            elif dtype in (np.int8,):
                test_input = np.random.randint(-128, 128, size=shape, dtype=np.int8)
            else:
                test_input = (
                    np.random.random(size=shape).astype(np.float32) * 1.0
                ).astype(dtype, copy=False)

            test_input = np.ascontiguousarray(test_input)

            log_model(f"Teste {mid} shape={shape} dtype={dtype} ...")

            for dev in devices:
                dev_key = _norm_dev_name(dev)
                if dev_key in model_row["benchmarks"]:
                    dev_key = str(dev)

                current_step += 1
                state.BENCHMARK_PROGRESS = min(
                    95, int((current_step / total_steps) * 95)
                )

                try:

                    def _bench_latency_sync(
                        model=model, dev=dev, test_input=test_input
                    ):
                        c_model = core.compile_model(
                            model, device_name=dev, config=_make_perf_config("LATENCY")
                        )
                        ireq = c_model.create_infer_request()

                        for _ in range(3):
                            ireq.infer([test_input])

                        times = []
                        t_start = time.perf_counter()
                        cnt = 0
                        while (time.perf_counter() - t_start) < lat_budget_s:
                            t0 = time.perf_counter()
                            ireq.infer([test_input])
                            times.append(time.perf_counter() - t0)
                            cnt += 1
                            if cnt >= max_iters:
                                break

                        if not times:
                            return 0.0
                        return float(np.mean(times) * 1000.0)

                    lat_ms = await asyncio.to_thread(_bench_latency_sync)
                except Exception:
                    lat_ms = 0.0

                current_step += 1
                state.BENCHMARK_PROGRESS = min(
                    95, int((current_step / total_steps) * 95)
                )

                try:

                    def _bench_throughput_sync(
                        *, dev=dev, model=model, test_input=test_input, core=core
                    ):
                        cfg = _make_perf_config("THROUGHPUT")

                        if _norm_dev_name(dev) == "GPU":
                            _cfg_set(
                                cfg,
                                properties.hint.allow_auto_batching,
                                "PERFORMANCE_HINT_ALLOW_AUTO_BATCHING",
                                False,
                            )
                            _cfg_set(
                                cfg,
                                properties.auto_batch_timeout,
                                "AUTO_BATCH_TIMEOUT",
                                0,
                            )

                        c_probe = core.compile_model(model, device_name=dev, config=cfg)
                        nireq = _get_optimal_nireq(c_probe)
                        nireq = max(1, min(int(nireq), int(max_nireq_cap)))

                        if _norm_dev_name(dev) == "GPU":
                            nireq = min(nireq, max(1, int(gpu_thr_cap)))

                        _cfg_set(
                            cfg,
                            properties.hint.num_requests,
                            "PERFORMANCE_HINT_NUM_REQUESTS",
                            str(nireq),
                        )

                        c_model = core.compile_model(model, device_name=dev, config=cfg)
                        r0 = c_model.create_infer_request()
                        for _ in range(3):
                            r0.infer({0: test_input})

                        done = 0
                        t0 = time.perf_counter()

                        try:
                            q = ov.AsyncInferQueue(c_model, nireq)

                            def _cb(infer_request, userdata):
                                nonlocal done
                                done += 1

                            q.set_callback(_cb)

                            while (time.perf_counter() - t0) < thr_budget_s:
                                q.start_async({0: test_input})

                            q.wait_all()
                            dt = max(1e-9, time.perf_counter() - t0)
                            fps = float(done / dt)
                            return fps, nireq, done, dt

                        except Exception:
                            cnt = 0
                            t0b = time.perf_counter()
                            while (time.perf_counter() - t0b) < thr_budget_s:
                                r0.infer({0: test_input})
                                cnt += 1
                                if cnt >= max_iters:
                                    break
                            dt = max(1e-9, time.perf_counter() - t0b)
                            fps = float(cnt / dt)
                            return fps, 1, cnt, dt

                    thr_fps, thr_nireq, thr_done, thr_dt = await asyncio.to_thread(
                        _bench_throughput_sync
                    )
                except Exception:
                    thr_fps, thr_nireq, thr_done, thr_dt = 0.0, 1, 0, 0.0

                lat_fps = (1000.0 / lat_ms) if lat_ms > 1e-9 else 0.0
                thr_ms = (1000.0 / thr_fps) if thr_fps > 1e-9 else 0.0

                model_row["benchmarks"][dev_key] = {
                    "lat_ms": round(float(lat_ms), 2),
                    "lat_fps": round(float(lat_fps), 1),
                    "thr_fps": round(float(thr_fps), 1),
                    "thr_ms": round(float(thr_ms), 2),
                    "thr_nireq": int(thr_nireq),
                    "thr_done": int(thr_done),
                    "thr_dt_s": round(float(thr_dt), 3),
                }

                with state.MODEL_LOCK:
                    tmp = list(matrix_results)
                    found = False
                    for i, r in enumerate(tmp):
                        if r.get("id") == mid:
                            tmp[i] = model_row
                            found = True
                            break
                    if not found:
                        tmp.append(model_row)
                    state.LAST_BENCHMARK_RESULTS = tmp

            matrix_results.append(model_row)

        except Exception as e:
            log_model(f"Fehler bei {mid}: {e}")
            matrix_results.append(model_row)

    log_model("=== Fertig ===")
    with state.MODEL_LOCK:
        state.LAST_BENCHMARK_RESULTS = list(matrix_results)
        state.BENCHMARK_PROGRESS = 100
        state.MODEL_RUNNING = False

    return matrix_results
