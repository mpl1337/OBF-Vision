import json
import logging
import threading
import time
from pathlib import Path

from app.utils import atomic_write_json

LOG = logging.getLogger("obf.stats")
STATS_FILE = Path("stats.json")
SAVE_INTERVAL = 5.0


class DetectionStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
        self.last_save = 0
        self._load()

    def _load(self):
        if STATS_FILE.exists():
            try:
                self.data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def update(self, name: str):
        name_s = str(name or "").strip()
        if not name_s or name_s.strip().lower() in (
            "unknown",
            "face",
            "blurry",
            "poorquality",
        ):
            return

        with self.lock:
            now = time.time()
            entry = self.data.get(name, {"count": 0, "last_seen": 0})

            entry["count"] += 1
            entry["last_seen"] = now
            self.data[name] = entry

            if now - self.last_save > SAVE_INTERVAL:
                self._flush()

    def _flush(self):
        try:
            atomic_write_json(STATS_FILE, self.data)
            self.last_save = time.time()
        except Exception:
            LOG.exception("Failed to write stats.json")

    def get_info(self, name: str):
        with self.lock:
            return self.data.get(name, None)


STATS = DetectionStats()
