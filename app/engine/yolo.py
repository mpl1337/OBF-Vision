"""
YOLOv11 Adapter for OpenVINO

This module uses logic and models based on Ultralytics YOLO11.
--------------------------------------------------------------------------
Software: Ultralytics YOLO11
Authors:  Glenn Jocher and Jing Qiu
Version:  11.0.0 (2024)
License:  AGPL-3.0
URL:      https://github.com/ultralytics/ultralytics

Citation:
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
--------------------------------------------------------------------------
"""

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.utils import clamp, nms_xyxy, resize_letterbox


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    label: str


class YoloOVAdapter:
    def __init__(
        self,
        ov,
        class_names: list[str],
        conf_thres: float = 0.35,
        iou_thres: float = 0.50,
        max_det: int = 100,
    ):
        self.ov = ov
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def detect(self, img_bgr: np.ndarray) -> tuple[list[Det], dict[str, Any]]:
        H, W = img_bgr.shape[:2]
        t0 = time.perf_counter()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        inp_img, sx, sy, px, py = resize_letterbox(img_rgb, self.ov.w, self.ov.h)
        inp = self.ov.prep_input(inp_img, normalize=True)

        out = self.ov.infer(inp)
        arr = next(iter(out.values()))

        dets, dbg = self._parse_outputs(arr, W, H, sx, sy, px, py)

        dt = (time.perf_counter() - t0) * 1000.0
        dbg.update({"time_ms": float(dt), "input_wh": [self.ov.w, self.ov.h]})
        return dets, dbg

    def _mk_det(
        self, x1: float, y1: float, x2: float, y2: float, conf: float, cls: int
    ) -> Det:
        label = (
            self.class_names[cls]
            if 0 <= cls < len(self.class_names)
            else f"class_{cls}"
        )
        return Det(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=cls, label=label)

    def _nms_limit(self, dets: list[Det]) -> list[Det]:
        if not dets:
            return []
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
        scores = np.array([d.conf for d in dets], dtype=np.float32)
        keep = nms_xyxy(boxes, scores, self.iou_thres)
        kept = [dets[i] for i in keep]
        kept.sort(key=lambda d: d.conf, reverse=True)
        return kept[: self.max_det]

    def _parse_outputs(
        self,
        arr: np.ndarray,
        orig_w: int,
        orig_h: int,
        sx: float,
        sy: float,
        px: int,
        py: int,
    ):
        a = np.array(arr)
        mat = a
        if mat.ndim == 3 and mat.shape[0] == 1:
            mat = mat[0]

        if mat.ndim == 2:
            rows = mat.T if mat.shape[0] < mat.shape[1] else mat
        elif mat.ndim == 3:
            rows = mat.reshape(mat.shape[0], -1).T
        else:
            return [], {"error": f"Bad shape {a.shape}"}

        N, K = rows.shape
        if K < 6:
            return [], {"error": "Rows too small"}

        coords = rows[:, :4]
        coord_max = float(np.max(coords)) if coords.size else 0.0
        normalized = coord_max <= 2.0

        dets: list[Det] = []
        for i in range(N):
            scores = rows[i, 4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])

            if conf < self.conf_thres:
                continue

            cx, cy, w, h = map(float, rows[i, :4])

            if normalized:
                cx *= self.ov.w
                w *= self.ov.w
                cy *= self.ov.h
                h *= self.ov.h

            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            x1 = (x1 - px) / (sx + 1e-12)
            y1 = (y1 - py) / (sy + 1e-12)
            x2 = (x2 - px) / (sx + 1e-12)
            y2 = (y2 - py) / (sy + 1e-12)

            x1 = clamp(x1, 0, orig_w - 1)
            y1 = clamp(y1, 0, orig_h - 1)
            x2 = clamp(x2, 0, orig_w - 1)
            y2 = clamp(y2, 0, orig_h - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            dets.append(self._mk_det(x1, y1, x2, y2, conf, cls))

        dets = self._nms_limit(dets)
        return dets, {"rows": N, "dets": len(dets)}
