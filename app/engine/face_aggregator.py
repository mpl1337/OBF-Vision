import time
from typing import Any

import numpy as np


class FaceAggregator:
    def __init__(self, max_embeds: int = 12):
        self.max_embeds = int(max_embeds)
        self._store: dict[Any, list[tuple[np.ndarray, float]]] = {}
        self._last_seen: dict[Any, float] = {}

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray | None:
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return None
        return v / (n + 1e-12)

    def add(self, key: Any, emb: np.ndarray, quality: float) -> bool:
        e = self._norm(emb)
        if e is None:
            return False
        lst = self._store.setdefault(key, [])
        self._last_seen[key] = time.time()
        lst.append((e, float(quality)))
        if len(lst) > self.max_embeds:
            self._store[key] = lst[-self.max_embeds :]
        return True

    def count(self, key: Any) -> int:
        return len(self._store.get(key, []))

    def mean(self, key: Any) -> np.ndarray | None:
        lst = self._store.get(key, [])
        if not lst:
            return None
        embs = np.stack([e for e, _q in lst], axis=0)
        avg = embs.mean(axis=0)
        return self._norm(avg)

    def clear(self, key: Any) -> None:
        self._store.pop(key, None)
        self._last_seen.pop(key, None)

    def prune(self, max_age_s: float = 90.0) -> int:
        try:
            max_age = float(max_age_s)
        except Exception:
            max_age = 90.0
        now = time.time()
        dead = [k for k, ts in self._last_seen.items() if (now - float(ts)) > max_age]
        for k in dead:
            self._store.pop(k, None)
            self._last_seen.pop(k, None)
        return len(dead)
