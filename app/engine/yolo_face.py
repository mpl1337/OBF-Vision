import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.utils import clamp, nms_xyxy, resize_letterbox

LOG = logging.getLogger("obf.yoloface")


@dataclass
class FaceDet:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    landmarks: np.ndarray | None


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class YoloFaceAdapter:
    def __init__(self, ov, conf_thres=0.45, iou_thres=0.40):
        self.ov = ov
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(
        self, img_bgr: np.ndarray, min_px: int = 20
    ) -> tuple[list[FaceDet], dict[str, Any]]:
        H, W = img_bgr.shape[:2]
        t0 = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp_img, sx, sy, px, py = resize_letterbox(img_rgb, self.ov.w, self.ov.h)
        inp = self.ov.prep_input(inp_img, normalize=True)
        out = self.ov.infer(inp)
        arr = next(iter(out.values()))[0]

        if arr.shape[0] < arr.shape[1]:
            arr = arr.T

        dets = []

        try:
            raw_scores = arr[:, 4]

            if np.max(raw_scores) > 1.0:
                scores = sigmoid(raw_scores)
            else:
                scores = raw_scores

            mask = scores > self.conf_thres
            candidates = arr[mask]
            final_scores = scores[mask]

            if candidates.shape[0] > 0:
                max_coord = np.max(candidates[:, 0])
                is_normalized = max_coord <= 1.5

                scale_w = self.ov.w if is_normalized else 1.0
                scale_h = self.ov.h if is_normalized else 1.0

                cx = candidates[:, 0] * scale_w
                cy = candidates[:, 1] * scale_h
                w = candidates[:, 2] * scale_w
                h = candidates[:, 3] * scale_h

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                nms_boxes = np.stack([x1, y1, x2, y2], axis=1)
                keep = nms_xyxy(nms_boxes, final_scores, self.iou_thres)

                has_landmarks = candidates.shape[1] >= 15

                for i in keep:
                    scr = float(final_scores[i])
                    b = nms_boxes[i]

                    bx1 = (b[0] - px) / sx
                    by1 = (b[1] - py) / sy
                    bx2 = (b[2] - px) / sx
                    by2 = (b[3] - py) / sy

                    bx1 = clamp(bx1, 0, W)
                    by1 = clamp(by1, 0, H)
                    bx2 = clamp(bx2, 0, W)
                    by2 = clamp(by2, 0, H)

                    if (bx2 - bx1) < min_px:
                        continue

                    kpts = None
                    if has_landmarks:
                        l_raw = candidates[i, 5:]

                        lm_list = []

                        for j in range(5):
                            step = 3 if candidates.shape[1] >= 20 else 2

                            idx_x = j * step
                            idx_y = j * step + 1

                            val_x = l_raw[idx_x]
                            val_y = l_raw[idx_y]

                            if is_normalized:
                                val_x *= scale_w
                                val_y *= scale_h

                            lx = (val_x - px) / sx
                            ly = (val_y - py) / sy
                            lm_list.append([lx, ly])

                        kpts = np.array(lm_list)

                    dets.append(FaceDet(bx1, by1, bx2, by2, scr, kpts))

        except Exception as e:
            LOG.error(f"YOLO Processing Error: {e}")

        dt = (time.perf_counter() - t0) * 1000
        return dets, {"time_ms": dt}
