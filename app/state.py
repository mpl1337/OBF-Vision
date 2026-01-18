import asyncio
import json
import threading
import time
from typing import Any

from fastapi import WebSocket

from .config import load_or_create_config
from .utils import LOG, PerfStats

STATE_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
ENROLL_LOCK = threading.Lock()

LAST_BENCHMARK_RESULTS = []
BENCHMARK_PROGRESS = 0

CFG: dict[str, Any] = load_or_create_config()
PIPE = None
PIPE_ENABLED: bool = bool(CFG.get("pipeline_enabled", True))

START_TIME = time.time()
PERF = PerfStats()

MAIN_LOOP: asyncio.AbstractEventLoop | None = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        with self.lock:
            conns = list(self.active_connections)

        for connection in conns:
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)


WS_MANAGER = ConnectionManager()

MODEL_LOG: list[str] = []
MODEL_RUNNING: bool = False

ENROLL_LOG: list[str] = []
ENROLL_RUNNING: bool = False


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    global MAIN_LOOP
    MAIN_LOOP = loop


def _fire_ws_json(msg_dict: dict[str, Any]):
    loop = globals().get("MAIN_LOOP", None)
    if loop is None or (hasattr(loop, "is_running") and not loop.is_running()):
        return

    try:
        asyncio.run_coroutine_threadsafe(
            WS_MANAGER.broadcast(json.dumps(msg_dict)), loop
        )
    except Exception:
        import logging as _logging
        try:
            from app.utils import log_exception_throttled_here
            log_exception_throttled_here(_logging.getLogger(__name__), "Suppressed exception (was 'pass')")
        except Exception:
            _logging.getLogger(__name__).exception("Suppressed exception (was 'pass') [throttle helper import failed]")


def log_model(msg: str) -> None:
    line = str(msg)
    with MODEL_LOCK:
        MODEL_LOG.append(line)
        if len(MODEL_LOG) > 800:
            MODEL_LOG[:] = MODEL_LOG[-800:]
        lines_copy = list(MODEL_LOG)
    _fire_ws_json({"type": "model_log", "lines": lines_copy})


def log_enroll(line: str) -> None:
    with ENROLL_LOCK:
        ENROLL_LOG.append(line)
        if len(ENROLL_LOG) > 800:
            ENROLL_LOG[:] = ENROLL_LOG[-800:]
        lines_copy = list(ENROLL_LOG)
    _fire_ws_json({"type": "enroll_log", "lines": lines_copy})


def stop_pipeline():
    global PIPE
    with STATE_LOCK:
        if PIPE is not None:
            try:
                if hasattr(PIPE, "shutdown"):
                    PIPE.shutdown()
            except Exception as e:
                LOG.error(f"Fehler beim Pipeline-Shutdown: {e}")
            PIPE = None
            LOG.info("Pipeline gestoppt.")
