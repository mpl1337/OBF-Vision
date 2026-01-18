from pathlib import Path

import cv2
import numpy as np

from .face_utils import align_face


class ArcFaceReID:
    def __init__(self, ov, preprocess: str = "auto"):
        self.ov = ov
        self.input_size = (int(getattr(ov, "w", 112)), int(getattr(ov, "h", 112)))

        raw_pre = (preprocess or "auto").lower().strip()
        self.preprocess = self._resolve_preprocess(raw_pre)

        mx = str(getattr(ov, "model_xml", "")).lower()
        self.align_template = "retail0095" if "retail-0095" in mx else "insightface"

    @staticmethod
    def _recommended_for_model(model_xml: str) -> str:
        name = (Path(model_xml).name if model_xml else "").lower()
        if "retail-0095" in name or "reidentification-retail" in name:
            return "raw"
        if "face-reidentification" in name and "0095" in name:
            return "raw"
        return "arcface"

    def _resolve_preprocess(self, preprocess: str) -> str:
        if preprocess in ("auto", ""):
            mx = str(getattr(self.ov, "model_xml", "")).lower()
            return self._recommended_for_model(mx)
        if preprocess in ("arcface", "insightface"):
            return "arcface"
        if preprocess in ("0_1", "01", "0-1"):
            return "0_1"
        return "raw"

    def _prep(self, img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = img_bgr[:, :, ::-1]
        x = img_rgb.astype(np.float32, copy=False)

        if self.preprocess == "arcface":
            x = (x - 127.5) / 127.5
        elif self.preprocess == "0_1":
            x = x / 255.0
        else:
            pass

        layout = str(getattr(self.ov, "layout", "NCHW")).upper()
        if layout == "NHWC":
            inp = x[None, ...]
        else:
            inp = np.transpose(x, (2, 0, 1))[None, ...]

        dt = getattr(self.ov, "input_np_dtype", np.float32)
        if dt in (np.float16, np.float32, np.float64):
            inp = inp.astype(dt, copy=False)
        return np.ascontiguousarray(inp)

    def embed(
        self, full_img_bgr: np.ndarray, landmarks: np.ndarray = None
    ) -> np.ndarray:
        w, h = self.input_size

        face_aligned = None

        if landmarks is not None:
            face_aligned = align_face(
                full_img_bgr,
                landmarks,
                out_size=(w, h),
                template=self.align_template,
            )

        if face_aligned is None:
            face_aligned = cv2.resize(full_img_bgr, (w, h))

        inp = self._prep(face_aligned)

        out = self.ov.infer(inp)
        emb = next(iter(out.values())).flatten()

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb
