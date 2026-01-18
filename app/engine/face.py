import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

LOG = logging.getLogger("obf.facedb")


@dataclass
class FaceMatch:
    name: str
    score: float
    index: int


class FaceDB:
    def __init__(self, path: str):
        self.path = Path(path)
        self.index = None
        self.names: dict[int, str] = {}
        self.dim: int | None = None
        self.metadata: dict[str, Any] = {}
        self.is_dirty = False

    def _paired_paths(self) -> tuple[Path, Path]:
        p = self.path
        sfx = p.suffix.lower()

        if sfx == ".json":
            return p.with_suffix(".faiss"), p
        if sfx == ".faiss":
            return p, p.with_suffix(".json")
        if sfx == "":
            return p.with_suffix(".faiss"), p.with_suffix(".json")

        return p, p.with_suffix(".json")

    def _is_supported_path(self) -> bool:
        sfx = self.path.suffix.lower()
        return sfx in ("", ".faiss", ".json")

    def load(self) -> None:
        if not self._is_supported_path():
            LOG.error(
                f"Unsupported FaceDB path extension: '{self.path.suffix}'. Use .faiss or .json."
            )
            self._create_empty()
            return

        idx_path, meta_path = self._paired_paths()

        if not idx_path.exists():
            LOG.warning(f"DB index missing: {idx_path}. Creating new (empty) DB.")
            self._create_empty()
            return

        try:
            self.index = faiss.read_index(str(idx_path))
            self.dim = int(getattr(self.index, "d", 0) or 0) or None

            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)

                names_raw = meta.get("names", {}) or {}
                self.names = {int(k): str(v) for k, v in names_raw.items()}
                self.metadata = meta.get("metadata", {}) or {}

                try:
                    meta_dim = (
                        int(meta.get("dim")) if meta.get("dim") is not None else None
                    )
                except Exception:
                    meta_dim = None

                try:
                    meta_ntotal = (
                        int(meta.get("ntotal"))
                        if meta.get("ntotal") is not None
                        else None
                    )
                except Exception:
                    meta_ntotal = None

                if meta_dim and self.dim and meta_dim != self.dim:
                    LOG.warning(
                        f"FaceDB meta dim mismatch: index={self.dim} vs meta={meta_dim} ({meta_path.name})"
                    )

                if meta_ntotal is not None:
                    try:
                        nt = int(self.index.ntotal)
                        if meta_ntotal != nt:
                            LOG.warning(
                                f"FaceDB meta ntotal mismatch: index={nt} vs meta={meta_ntotal} ({meta_path.name})"
                            )
                    except Exception:
                        pass
            else:
                self.names = {}
                self.metadata = {}
                LOG.warning(f"Meta JSON missing: {meta_path}. Names/metadata empty.")

            mod_tag = self.metadata.get("model_tag", "unknown")
            LOG.info(
                f"FaceDB loaded ({len(self.names)} ids, dim={self.dim}, model={mod_tag}) via FAISS+JSON."
            )

        except Exception as e:
            LOG.error(f"Failed to load DB: {e}. Starting empty.")
            self._create_empty()

    def _create_empty(self):
        self.index = None
        self.dim = None
        self.names = {}
        self.metadata = {}


    def _atomic_replace(self, src: Path, dst: Path, retries: int = 10) -> None:
        last_err = None
        for i in range(retries):
            try:
                os.replace(src, dst)
                last_err = None
                break
            except OSError as e:
                last_err = e
                time.sleep(0.05 * (i + 1))
        if last_err is not None:
            raise last_err

    def _write_json_atomic(self, dst: Path, payload: dict[str, Any]) -> None:
        tmp = dst.with_name(dst.name + f".tmp.{uuid.uuid4().hex}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        self._atomic_replace(tmp, dst)

    def _write_faiss_atomic(self, dst: Path, index_obj) -> None:
        tmp = dst.with_name(dst.name + f".tmp.{uuid.uuid4().hex}")
        faiss.write_index(index_obj, str(tmp))
        try:
            with open(tmp, "rb") as f:
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            pass
        self._atomic_replace(tmp, dst)

    def save(self):
        if not self.path:
            return

        if not self._is_supported_path():
            LOG.error(
                f"Unsupported FaceDB path extension: '{self.path.suffix}'. Use .faiss or .json."
            )
            return

        idx_path, meta_path = self._paired_paths()

        try:
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            index_obj = self.index
            if index_obj is None:
                if self.dim is None:
                    meta = {
                        "version": "v3-faiss+json",
                        "dim": None,
                        "ntotal": 0,
                        "names": {},
                        "metadata": self.metadata,
                    }
                    self._write_json_atomic(meta_path, meta)
                    self.is_dirty = False
                    LOG.info("FaceDB saved meta only (empty, dim unknown).")
                    return
                index_obj = faiss.IndexFlatIP(int(self.dim))

            self._write_faiss_atomic(idx_path, index_obj)

            meta = {
                "version": "v3-faiss+json",
                "dim": int(self.dim) if self.dim is not None else None,
                "ntotal": (
                    int(index_obj.ntotal) if hasattr(index_obj, "ntotal") else None
                ),
                "names": {str(k): v for k, v in (self.names or {}).items()},
                "metadata": self.metadata,
            }
            self._write_json_atomic(meta_path, meta)

            self.is_dirty = False
            LOG.info(f"FaceDB saved (FAISS+JSON): {idx_path.name} + {meta_path.name}")

        except Exception as e:
            LOG.error(f"Failed to save DB: {e}")

    def add_person(self, name: str, embedding: np.ndarray):
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        d = int(emb.shape[1])

        if self.index is None:
            self.dim = d
            self.index = faiss.IndexFlatIP(self.dim)
            LOG.info(f"FaceDB initialized with dimension: {self.dim}")

        if self.dim != d:
            LOG.error(
                f"Vector dimension mismatch! DB expects {self.dim}, got {d}. Skipping {name}."
            )
            return

        faiss.normalize_L2(emb)
        self.index.add(emb)
        new_id = int(self.index.ntotal - 1)
        self.names[new_id] = str(name)
        self.is_dirty = True

    def best_match(self, emb: np.ndarray) -> FaceMatch | None:
        if self.index is None or self.index.ntotal == 0:
            return None
        if emb is None:
            return None

        e = np.asarray(emb)
        if e.ndim == 2:
            if e.shape[0] != 1:
                return None
            e = e.reshape(-1)
        elif e.ndim != 1:
            return None

        if self.dim is None or e.shape[0] != self.dim:
            return None

        q = e.reshape(1, -1).astype(np.float32, copy=False)
        faiss.normalize_L2(q)

        dists, idxs = self.index.search(q, 1)
        idx = int(idxs[0][0])
        if idx < 0:
            return None

        score = float(dists[0][0])
        name = self.names.get(idx, "Unknown")
        return FaceMatch(name=name, score=score, index=idx)
