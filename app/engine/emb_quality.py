import numpy as np


def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def filter_outliers_to_centroid(
    embs: np.ndarray, keep_quantile: float = 0.25
) -> np.ndarray:
    if embs is None or len(embs) == 0:
        return embs
    E = l2norm(embs)
    c = l2norm(E.mean(axis=0))
    sims = (E @ c.T).reshape(-1)
    cutoff = np.quantile(sims, keep_quantile)
    keep = sims >= cutoff
    if keep.sum() == 0:
        keep[np.argmax(sims)] = True
    return E[keep]


def dedupe_greedy(embs: np.ndarray, sim_th: float = 0.995) -> np.ndarray:
    if embs is None or len(embs) <= 1:
        return embs
    E = l2norm(embs)
    kept = []
    for e in E:
        if not kept:
            kept.append(e)
            continue
        K = np.stack(kept, axis=0)
        sims = (K @ e.reshape(-1, 1)).reshape(-1)
        if np.max(sims) < sim_th:
            kept.append(e)
    return np.stack(kept, axis=0)


def pick_prototypes(E: np.ndarray, k: int = 5) -> np.ndarray:
    if E is None or len(E) == 0:
        return E
    E = l2norm(E)
    if len(E) <= k:
        return E

    c = l2norm(E.mean(axis=0))
    sims = (E @ c.T).reshape(-1)
    idx0 = int(np.argmax(sims))
    chosen = [idx0]

    while len(chosen) < k:
        K = E[chosen]
        max_sim = (E @ K.T).max(axis=1)
        dist = 1.0 - max_sim
        dist[chosen] = -1
        chosen.append(int(np.argmax(dist)))

    return E[chosen]
