import cv2
import numpy as np

_INSIGHTFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

INSIGHTFACE_REF_NORM = _INSIGHTFACE_REF_112 / 112.0

RETAIL0095_REF_NORM = np.array(
    [
        [0.31556875, 0.4615741071428571],
        [0.6826229166666667, 0.4615741071428571],
        [0.5002625, 0.6405053571428571],
        [0.34947187500000004, 0.8246919642857142],
        [0.6534364583333333, 0.8246919642857142],
    ],
    dtype=np.float32,
)


def _ref_points(template: str, out_w: int, out_h: int) -> np.ndarray:
    t = (template or "insightface").lower()
    if t in ("retail0095", "retail-0095", "0095"):
        ref = RETAIL0095_REF_NORM
    else:
        ref = INSIGHTFACE_REF_NORM

    pts = ref.copy()
    pts[:, 0] *= float(out_w)
    pts[:, 1] *= float(out_h)
    return pts.astype(np.float32)


def _normalize_landmarks(landmarks) -> np.ndarray | None:
    if landmarks is None:
        return None
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.size == 10:
        lm = lm.reshape(5, 2)
    if lm.shape != (5, 2):
        return None
    return lm


def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    out_size=(112, 112),
    template: str = "insightface",
) -> np.ndarray | None:
    out_w, out_h = int(out_size[0]), int(out_size[1])

    lm = _normalize_landmarks(landmarks)
    if lm is None:
        return None

    ref = _ref_points(template, out_w, out_h)

    try:
        M, inliers = cv2.estimateAffinePartial2D(lm, ref)
    except Exception:
        M = None

    if M is None:
        return None

    aligned = cv2.warpAffine(img, M, (out_w, out_h), borderValue=(0, 0, 0))
    return aligned


def check_face_quality(landmarks: np.ndarray) -> float:
    if landmarks is None or landmarks.size < 10:
        return 0.0

    lm = landmarks.reshape(5, 2)

    eye_l = lm[0]
    eye_r = lm[1]
    nose = lm[2]

    d_l = np.linalg.norm(eye_l - nose)
    d_r = np.linalg.norm(eye_r - nose)

    ratio = min(d_l, d_r) / (max(d_l, d_r) + 1e-6)

    if ratio > 0.5:
        return 1.0
    return ratio / 0.5
