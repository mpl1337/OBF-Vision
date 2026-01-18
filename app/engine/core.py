import logging
import os
import queue
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
from openvino import Core, properties

LOG = logging.getLogger("obf.core")

CACHE_DIR = (
    Path(os.getenv("OBF_OV_CACHE_DIR", "./model_cache")).resolve()
    / f"ov_{ov.__version__}"
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    GLOBAL_CORE = Core()
    try:
        try:
            cache_key = properties.cache_dir()
        except Exception:
            cache_key = "CACHE_DIR"
        GLOBAL_CORE.set_property({cache_key: str(CACHE_DIR)})
        LOG.info(f"OpenVINO cache dir: {CACHE_DIR}")
    except Exception as e:
        LOG.warning(f"Could not set OpenVINO cache dir: {e}")

    devices = GLOBAL_CORE.available_devices
    LOG.info(f"OpenVINO Core init. Devices: {devices}")

    if "NPU" in devices:
        LOG.info("Intel NPU erkannt! Aktiviere High-Performance Config.")
except Exception as e:
    LOG.error(f"Failed to init OpenVINO Core: {e}")
    GLOBAL_CORE = None


class OVModel:
    def __init__(
        self,
        core: Core,
        model_xml: str,
        device: str,
        num_requests: int = 0,
        perf_hint: str = "LATENCY",
    ):
        self.core = GLOBAL_CORE if GLOBAL_CORE else core
        self.model_xml = model_xml
        self.device = str(device)
        self.perf_hint = str(perf_hint or "LATENCY").upper().strip() or "LATENCY"

        if self.device.upper() == "NPU" and "NPU" not in self.core.available_devices:
            LOG.warning("NPU ausgewählt, aber nicht verfügbar. Fallback auf CPU.")
            self.device = "CPU"

        LOG.info(
            f"Loading model: {Path(model_xml).name} on {self.device} (Hint: {self.perf_hint})..."
        )
        self.model = self.core.read_model(model=model_xml)

        config = {}

        if self.device.upper() in ["NPU", "GPU"]:
            mode = self.perf_hint
            try:
                val = (
                    properties.hint.PerformanceMode.THROUGHPUT
                    if mode == "THROUGHPUT"
                    else properties.hint.PerformanceMode.LATENCY
                )
                config[properties.hint.performance_mode()] = val
            except Exception:
                config["PERFORMANCE_HINT"] = mode

        try:
            self.compiled = self.core.compile_model(
                self.model, device_name=self.device, config=config
            )
        except RuntimeError as e:
            LOG.error(f"Compilation failed on {self.device}. Error: {e}")
            raise

        self.input_layer = self.compiled.inputs[0]
        self.output_layers = list(self.compiled.outputs)

        self.n, self.c, self.h, self.w = 1, 3, 640, 640
        self.layout = "NCHW"
        self.input_shape = None

        try:
            ps = self.input_layer.get_partial_shape()
            dims = []
            for i, d in enumerate(ps):
                try:
                    is_dyn = bool(getattr(d, "is_dynamic", False))
                except Exception:
                    is_dyn = False

                if is_dyn:
                    if i == 0:
                        dims.append(1)
                    elif i == 1:
                        dims.append(3)
                    else:
                        dims.append(640)
                else:
                    try:
                        dims.append(int(d.get_length()))
                    except Exception:
                        dims.append(int(d))

            if len(dims) == 4:
                if int(dims[1]) == 3:
                    self.layout = "NCHW"
                    self.n, self.c, self.h, self.w = map(int, dims)
                    self.input_shape = tuple(map(int, dims))
                elif int(dims[3]) == 3:
                    self.layout = "NHWC"
                    n, h, w, c = map(int, dims)
                    self.n, self.c, self.h, self.w = n, c, h, w
                    self.input_shape = tuple(map(int, dims))
                else:
                    self.layout = "NCHW"
                    self.n, self.c, self.h, self.w = map(int, dims)
                    self.input_shape = tuple(map(int, dims))
        except Exception:
            pass

        LOG.info(
            f"Resolved model input: layout={self.layout}, raw={self.input_shape} "
            f"(as NCHW={self.n}x{self.c}x{self.h}x{self.w}) for {Path(model_xml).name}"
        )

        et = str(self.input_layer.element_type).lower()
        self.input_np_dtype = {
            "u8": np.uint8,
            "i8": np.int8,
            "f16": np.float16,
            "f32": np.float32,
        }.get(et, np.float32)

        self.optimal_nireq = self._get_optimal_number_of_infer_requests()
        self.pool_max = self._select_pool_size(num_requests)

        self.request_pool = queue.Queue(maxsize=max(0, int(self.pool_max) or 0))
        for _ in range(int(self.pool_max)):
            self.request_pool.put(self.compiled.create_infer_request())

        LOG.info(
            f"Initialized Request Pool: selected={self.pool_max} (opt={self.optimal_nireq}, hint={self.perf_hint}) "
            f"for {Path(model_xml).name}"
        )

        self._warmup()

    def _get_optimal_number_of_infer_requests(self) -> int | None:
        try:
            v = self.compiled.get_property(properties.optimal_number_of_infer_requests)
            iv = int(v)
            return iv if iv > 0 else None
        except Exception:
            pass

        try:
            v = self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            iv = int(v)
            return iv if iv > 0 else None
        except Exception:
            return None

    def _select_pool_size(self, num_requests: int) -> int:
        cap = int(os.getenv("OBF_OV_POOL_CAP", "32"))
        floor = int(os.getenv("OBF_OV_POOL_MIN", "1"))
        cap = max(1, cap)
        floor = max(1, floor)

        if int(num_requests) > 0:
            chosen = int(num_requests)
        else:
            if self.perf_hint == "THROUGHPUT":
                chosen = int(self.optimal_nireq or 4)
            else:
                chosen = 1

        chosen = max(floor, chosen)
        chosen = min(cap, chosen)
        return int(chosen)

    def _warmup(self):
        try:
            n = 1
            c = int(self.c or 3)
            h = int(self.h or 640)
            w = int(self.w or 640)

            if getattr(self, "layout", "NCHW") == "NHWC":
                dummy_input = np.zeros((n, h, w, c), dtype=self.input_np_dtype)
            else:
                dummy_input = np.zeros((n, c, h, w), dtype=self.input_np_dtype)

            self.infer(dummy_input)
        except Exception as e:
            LOG.warning(f"Warmup failed: {e}")

    def prep_input(self, img_hwc: np.ndarray, normalize: bool = True) -> np.ndarray:
        dt = getattr(self, "input_np_dtype", np.float32)

        if dt in (np.float16, np.float32, np.float64):
            x = img_hwc.astype(np.float32, copy=False)
            if normalize:
                x = x / 255.0
            if dt != np.float32:
                x = x.astype(dt, copy=False)
        else:
            x = img_hwc.astype(dt, copy=False)

        if getattr(self, "layout", "NCHW") == "NHWC":
            x = x[None, ...]
        else:
            x = np.transpose(x, (2, 0, 1))[None, ...]

        return np.ascontiguousarray(x)

    def infer(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        req: Any | None = None
        borrowed_from_pool = False

        try:
            try:
                req = self.request_pool.get(timeout=5.0)
                borrowed_from_pool = True
            except queue.Empty:
                LOG.warning(
                    f"InferRequest-Pool erschöpft (timeout). Erzeuge temporären Request."
                )
                req = self.compiled.create_infer_request()
                borrowed_from_pool = False

            req.infer({self.input_layer: input_tensor})

            out: dict[str, np.ndarray] = {}
            for i, o in enumerate(self.output_layers):
                try:
                    name = o.get_any_name()
                except Exception:
                    name = f"output_{i}"
                out[name] = req.get_tensor(o).data.copy()

            if borrowed_from_pool:
                try:
                    self.request_pool.put_nowait(req)
                except queue.Full:
                    LOG.error("Pool voll beim Zurückgeben (sollte nicht passieren!)")

            return out

        except Exception as e:
            if borrowed_from_pool and req is not None:
                try:
                    fresh_req = self.compiled.create_infer_request()
                    self.request_pool.put_nowait(fresh_req)
                except Exception as repair_err:
                    LOG.error(f"Pool repair failed: {repair_err}")

            LOG.exception("Inference failed")
            raise

    def get_pool_info(self) -> dict[str, Any]:
        model_file = Path(getattr(self, "model_xml", "")).name
        device = str(getattr(self, "device", "") or "")
        hint = str(getattr(self, "perf_hint", "") or "")

        selected = None
        free = None
        qmax = None

        try:
            selected = int(getattr(self, "pool_max", 0) or 0) or None
        except Exception:
            selected = None

        try:
            q = getattr(self, "request_pool", None)
            if q is not None:
                free = int(q.qsize())
                qmax = int(getattr(q, "maxsize", 0) or 0)
                if selected is None and qmax and qmax > 0:
                    selected = qmax
        except Exception:
            pass

        opt = None
        try:
            opt = int(
                self.compiled.get_property(properties.optimal_number_of_infer_requests)
            )
        except Exception:
            try:
                opt = int(
                    self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
                )
            except Exception:
                opt = None

        return {
            "model": model_file,
            "device": device,
            "hint": hint,
            "pool_selected": selected,
            "pool_optimal": opt,
            "pool_free": free,
            "pool_queue_max": qmax,
        }
