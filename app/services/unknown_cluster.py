import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


@dataclass
class UnknownItem:
    file: str
    score: float
    source: str
    suffix: str
    emb: np.ndarray


class UnknownClusterService:
    def __init__(self, unknown_dir: str):
        self.unknown_dir = Path(unknown_dir)

    def load_items(self, limit: int = 2000) -> tuple[list[UnknownItem], list[str]]:
        items: list[UnknownItem] = []
        missing: list[str] = []
        if not self.unknown_dir.exists():
            return items, missing

        imgs = sorted(
            [
                p
                for p in self.unknown_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ],
            key=lambda p: p.name,
            reverse=True,
        )

        for img in imgs[: int(limit)]:
            base_name = img.name.rsplit(".", 1)[0]
            base = img.parent / base_name

            npy = Path(str(base) + ".npy")
            js = Path(str(base) + ".json")

            if not npy.exists():
                missing.append(img.name)
                continue

            try:
                emb = np.load(npy).astype(np.float32, copy=False)
                emb = _l2norm(emb)
            except Exception:
                missing.append(img.name)
                continue

            meta = {"score": 0.0, "source": "Unknown", "suffix": "raw"}
            if js.exists():
                try:
                    meta.update(json.loads(js.read_text(encoding="utf-8")))
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
                            ".\\app\\services\\unknown_cluster.py:74",
                            "Suppressed exception (was 'pass')",
                        )

            items.append(
                UnknownItem(
                    file=img.name,
                    score=float(meta.get("score", 0.0) or 0.0),
                    source=str(meta.get("source", "Unknown")),
                    suffix=str(meta.get("suffix", "raw")),
                    emb=emb,
                )
            )

        return items, missing

    @staticmethod
    def _union_find(n: int):
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        return find, union

    def cluster(
        self,
        items: list[UnknownItem],
        sim_thres: float = 0.75,
        min_cluster_size: int = 2,
    ) -> list[dict[str, Any]]:
        n = len(items)
        if n == 0:
            return []

        shape_groups: dict[tuple[int, ...], list[int]] = {}
        for i, it in enumerate(items):
            try:
                shape_groups.setdefault(tuple(it.emb.shape), []).append(i)
            except Exception:
                continue

        if not shape_groups:
            return []

        _best_shape, best_idxs = max(shape_groups.items(), key=lambda kv: len(kv[1]))

        if len(best_idxs) != n:
            items = [items[i] for i in best_idxs]
            n = len(items)

        E = np.stack([it.emb for it in items]).astype(np.float32, copy=False)

        find, union = self._union_find(n)

        window = min(250, n)
        th = float(sim_thres)

        for i in range(n):
            jmax = min(n, i + window)
            if i + 1 >= jmax:
                continue
            vi = E[i]
            sims = E[i + 1 : jmax] @ vi
            hit = np.where(sims >= th)[0]
            for h in hit:
                union(i, i + 1 + int(h))

        groups: dict[int, list[int]] = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        clusters: list[dict[str, Any]] = []
        for _, idxs in groups.items():
            if len(idxs) < int(min_cluster_size):
                continue
            idxs_sorted = sorted(idxs)
            rep = idxs_sorted[0]
            clusters.append(
                {
                    "id": int(rep),
                    "count": int(len(idxs)),
                    "rep_file": items[rep].file,
                    "source": items[rep].source,
                    "files": [items[i].file for i in idxs_sorted],
                }
            )

        clusters.sort(key=lambda c: (c["count"], c["rep_file"]), reverse=True)
        return clusters
