import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import format_datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, List
from inspect import currentframe

import cv2
import numpy as np
from PIL import Image

_EXC_THROTTLE_LAST: dict[str, float] = {}


def log_exception_throttled(
    logger: logging.Logger, key: str, msg: str, interval_s: float = 30.0
) -> None:
    try:
        import sys as _sys

        exc = _sys.exc_info()[1]
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            return
        try:
            import asyncio as _asyncio

            if isinstance(exc, _asyncio.CancelledError):
                return
        except Exception:
            pass
        now = time.time()
        last = float(_EXC_THROTTLE_LAST.get(key, 0.0))
        if (now - last) < float(interval_s):
            return
        _EXC_THROTTLE_LAST[key] = now
        logger.exception(msg)
    except Exception:
        return
    
def log_exception_throttled_here(
    logger: logging.Logger,
    msg: str,
    interval_s: float = 30.0,
    key: str | None = None,
    stacklevel: int = 2,
) -> None:
    if key is None:
        fr = currentframe()
        for _ in range(max(0, int(stacklevel))):
            fr = fr.f_back if fr and fr.f_back else fr
        if fr and fr.f_code:
            fn = Path(fr.f_code.co_filename).name
            key = f"{fn}:{fr.f_lineno}:{msg}"
        else:
            key = msg
    log_exception_throttled(logger, key, msg, interval_s)    


class _UvicornNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("uvicorn."):
            record.name = "uvicorn"
        return True


LOG = logging.getLogger("obf")
_LOGGING_CONFIGURED = False


def setup_logging(
    level: str = "INFO", log_to_file: bool = False, max_mb: int = 5
) -> None:
    global _LOGGING_CONFIGURED
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not _LOGGING_CONFIGURED:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.addFilter(_UvicornNameFilter())
        root.addHandler(ch)

        if log_to_file:
            try:
                fh = RotatingFileHandler(
                    "obf_app.log",
                    maxBytes=max_mb * 1024 * 1024,
                    backupCount=3,
                    encoding="utf-8",
                )
                fh.setFormatter(fmt)
                fh.addFilter(_UvicornNameFilter())
                root.addHandler(fh)
            except Exception as e:
                print(f"ERROR enabling file logging: {e}")

        for n in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            lg = logging.getLogger(n)
            lg.handlers.clear()
            lg.propagate = True
        _LOGGING_CONFIGURED = True


def calc_blur_score(img_bgr: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 0.0


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def norm_ov_device_name(d: str) -> str:
    s = str(d or "").strip()
    if not s:
        return s
    base = s.split(":")[0].split(".")[0].strip()
    up = base.upper()
    return up if up in ("CPU", "GPU", "NPU") else base


def hard_exit(code: int = 0) -> None:
    try:
        import logging

        logging.shutdown()
    except Exception:
        pass
    try:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    import os

    os._exit(int(code))


def decode_image_bytes(img_bytes: bytes) -> np.ndarray:
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    import io

    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(im, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def resize_letterbox(
    img_bgr: np.ndarray, new_w: int, new_h: int, color=(114, 114, 114)
):
    h, w = img_bgr.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((new_h, new_w, 3), dtype=np.uint8), 1.0, 1.0, 0, 0
    r = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * r)), int(round(h * r))
    try:
        resized = cv2.resize(
            img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
    except Exception:
        resized = np.zeros((resized_h, resized_w, 3), dtype=np.uint8)
    pad_x = (new_w - resized_w) // 2
    pad_y = (new_h - resized_h) // 2
    out = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    out[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return out, r, r, pad_x, pad_y


def to_nchw(img_bgr: np.ndarray, normalize: bool, dtype: np.dtype) -> np.ndarray:
    if dtype in (np.float16, np.float32, np.float64):
        x = img_bgr.astype(np.float32, copy=False)
        if normalize:
            x = x / 255.0
        x = x.astype(dtype, copy=False) if dtype != np.float32 else x
    else:
        x = img_bgr.astype(dtype, copy=False)

    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(x)


def iou_xyxy(a, b) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2] - x1
    h = boxes[:, 3] - y1

    rects = np.column_stack((x1, y1, w, h)).tolist()
    sc = scores.tolist()

    indices = cv2.dnn.NMSBoxes(rects, sc, 0.0, iou_th)

    if len(indices) == 0:
        return []
    return [int(i) for i in indices.flatten()]


def utc_rfc1123() -> str:
    return format_datetime(datetime.now(UTC), usegmt=True)


def atomic_write_json(path: Path, data: dict[str, Any], indent: int = 2) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + f".tmp.{uuid.uuid4().hex}")
    try:
        payload = json.dumps(data, indent=indent, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(str(tmp_path), str(dest))
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def draw_predictions_on_image(
    img_bgr: np.ndarray, preds: list[dict[str, Any]]
) -> np.ndarray:
    out = img_bgr.copy()
    for p in preds:
        x1, y1 = int(p["x_min"]), int(p["y_min"])
        x2, y2 = int(p["x_max"]), int(p["y_max"])
        conf = float(p.get("confidence", 0.0) or 0.0)
        label_raw = str(p.get("label", "") or "")
        label_norm = label_raw.strip().lower()
        uid = p.get("userid")
        has_uid = bool(str(uid).strip() if uid is not None else "")

        color = (255, 0, 0)
        if label_norm == "person" or label_norm.startswith("person"):
            color = (0, 255, 255)
        elif has_uid:
            color = (0, 0, 255)
        elif (
            label_norm in ("face", "unknown")
            or label_norm.startswith("face")
            or label_norm.startswith("unknown")
        ):
            color = (0, 0, 255)
        elif label_norm == "blurry":
            color = (128, 128, 128)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        shown = uid if has_uid else label_raw
        text = f"{shown} {conf:.0%}"
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        c2 = (x1 + t_size[0] + 3, y1 - t_size[1] - 4)
        cv2.rectangle(out, (x1, y1), c2, color, -1)
        cv2.putText(
            out, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
    return out


_PERF_LOCK = threading.Lock()


@dataclass
class PerfStats:
    req_count: int = 0
    ok_count: int = 0
    fail_count: int = 0
    last_ms: float = 0.0
    avg_ms: float = 0.0
    ema_ms: float = 0.0
    ema_alpha: float = 0.10
    last_labels: List[str] = field(default_factory=list)

    def update(self, ok: bool, ms: float, labels: list[str]) -> None:
        with _PERF_LOCK:
            self.req_count += 1
            if ok:
                self.ok_count += 1
            else:
                self.fail_count += 1
            self.last_ms = float(ms)
            if self.req_count == 1:
                self.avg_ms = float(ms)
                self.ema_ms = float(ms)
            else:
                self.avg_ms = (
                    self.avg_ms * (self.req_count - 1) + float(ms)
                ) / self.req_count
                self.ema_ms = (
                    1.0 - self.ema_alpha
                ) * self.ema_ms + self.ema_alpha * float(ms)
            self.last_labels = labels[:20] if labels else []

    def snapshot(self) -> dict[str, Any]:
        with _PERF_LOCK:
            return {
                "req_count": int(self.req_count),
                "ok_count": int(self.ok_count),
                "fail_count": int(self.fail_count),
                "last_ms": float(self.last_ms),
                "avg_ms": float(self.avg_ms),
                "ema_ms": float(self.ema_ms),
                "ema_alpha": float(self.ema_alpha),
                "last_labels": list(self.last_labels or []),
            }
