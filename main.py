import asyncio
import faulthandler
import logging
import os
import signal
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from app.depcheck import assert_runtime_deps

    assert_runtime_deps(extra_imports=["pydantic"])
except SystemExit:
    raise
except Exception:
    print("KRITISCHER START-FEHLER (Dependency-Check):")
    traceback.print_exc()
    sys.exit(2)

import atexit

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_FH = open("obf_hang_traces.log", "a", buffering=1, encoding="utf-8")
faulthandler.enable(file=_FH, all_threads=True)


def _cleanup_faulthandler():
    try:
        _FH.flush()
        _FH.close()
    except:
        pass


atexit.register(_cleanup_faulthandler)

if hasattr(signal, "SIGBREAK"):
    if hasattr(faulthandler, "register"):
        try:
            faulthandler.register(signal.SIGBREAK, file=_FH, all_threads=True)
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
                    ".\\main.py:29",
                    "Suppressed exception (was 'pass')",
                )
    else:

        def _on_sigbreak(signum, frame):
            try:
                faulthandler.dump_traceback(file=_FH, all_threads=True)
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
                        ".\\main.py:36",
                        "Suppressed exception (was 'pass')",
                    )

        try:
            signal.signal(signal.SIGBREAK, _on_sigbreak)
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
                    ".\\main.py:41",
                    "Suppressed exception (was 'pass')",
                )

try:
    faulthandler.cancel_dump_traceback_later()
except Exception:
    pass

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


try:
    import app.state as state
    from app.engine.pipeline import ensure_pipeline_loaded
    from app.routers import gui_api, vision
    from app.state import CFG
    from app.utils import setup_logging
except Exception:
    print("KRITISCHER START-FEHLER (Imports):")
    traceback.print_exc()
    sys.exit(1)


LOG_LEVEL_STR = str(CFG.get("loglevel", "INFO")).upper()
LOG_TO_FILE = bool(CFG.get("log_rotate_enable", False))
LOG_MAX_MB = int(CFG.get("log_rotate_mb", 5))

setup_logging(LOG_LEVEL_STR, log_to_file=LOG_TO_FILE, max_mb=LOG_MAX_MB)
LOG = logging.getLogger("obf.main")

HEARTBEAT_FILE = Path(".obf_heartbeat")
HEARTBEAT_INTERVAL_SEC = 1.0

LOOPBEAT_FILE = Path(".obf_loopbeat")
LOOPBEAT_INTERVAL_SEC = 1.0


def _heartbeat_worker(stop_evt: threading.Event):
    while not stop_evt.is_set():
        try:
            HEARTBEAT_FILE.write_text(str(time.time()), encoding="utf-8")
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
                    ".\\main.py:86",
                    "Suppressed exception (was 'pass')",
                )
        stop_evt.wait(HEARTBEAT_INTERVAL_SEC)


async def _loopbeat_worker(stop_evt: asyncio.Event):
    while not stop_evt.is_set():
        try:
            LOOPBEAT_FILE.write_text(str(time.time()), encoding="utf-8")
        except Exception:
            import logging as _logging

            try:
                from app.utils import log_exception_throttled
            except Exception:
                _logging.getLogger(__name__).exception(
                    "Loopbeat worker error [throttle helper import failed]"
                )
            else:
                log_exception_throttled(
                    _logging.getLogger(__name__),
                    r".\main.py:_loopbeat_worker",
                    "Loopbeat worker error",
                )
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=LOOPBEAT_INTERVAL_SEC)
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        except Exception:
            import logging as _logging

            try:
                from app.utils import log_exception_throttled
            except Exception:
                _logging.getLogger(__name__).exception(
                    "Loopbeat worker error [throttle helper import failed]"
                )
            else:
                log_exception_throttled(
                    _logging.getLogger(__name__),
                    r".\main.py:_loopbeat_worker",
                    "Loopbeat worker error",
                )


HANG_MONITOR_ENABLE = int(os.getenv("OBF_HANG_MONITOR", "1") or "1")
HANG_STARTUP_GRACE_S = float(os.getenv("OBF_HANG_STARTUP_GRACE_S", "30") or "30")
HANG_AFTER_S = float(os.getenv("OBF_HANG_AFTER_S", "20") or "20")
HANG_POLL_S = float(os.getenv("OBF_HANG_POLL_S", "2.0") or "2.0")
HANG_COOLDOWN_S = float(os.getenv("OBF_HANG_COOLDOWN_S", "300") or "300")
HANG_MAX_DUMPS = int(os.getenv("OBF_HANG_MAX_DUMPS", "3") or "3")


def _safe_age_s(p: Path) -> float | None:
    try:
        if not p.exists():
            return None
        return max(0.0, time.time() - float(p.stat().st_mtime))
    except Exception:
        return None


def _hang_monitor_worker(stop_evt: threading.Event):
    start_ts = time.time()
    last_dump_ts = 0.0
    dumps = 0

    while not stop_evt.is_set():
        stop_evt.wait(HANG_POLL_S)
        if stop_evt.is_set():
            break

        now = time.time()
        if (now - start_ts) < HANG_STARTUP_GRACE_S:
            continue

        age_lb = _safe_age_s(LOOPBEAT_FILE)
        age_hb = _safe_age_s(HEARTBEAT_FILE)

        loop_stale = (age_lb is None) or (age_lb > HANG_AFTER_S)

        if not loop_stale:
            continue

        if dumps >= HANG_MAX_DUMPS:
            continue
        if (now - last_dump_ts) < HANG_COOLDOWN_S:
            continue

        dumps += 1
        last_dump_ts = now

        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            _FH.write("\n" + "=" * 72 + "\n")
            _FH.write(f"[HANG_MONITOR] {ts}\n")
            _FH.write(f"  loopbeat_age_s: {age_lb}\n")
            _FH.write(f"  heartbeat_age_s: {age_hb}\n")
            _FH.write(
                f"  thresholds: HANG_AFTER_S={HANG_AFTER_S}  cooldown={HANG_COOLDOWN_S}  max={HANG_MAX_DUMPS}\n"
            )
            _FH.write("=" * 72 + "\n")
            _FH.flush()
        except Exception:
            pass

        try:
            faulthandler.dump_traceback(file=_FH, all_threads=True)
            try:
                _FH.flush()
            except Exception:
                pass
        except Exception:
            pass


def _file_age_info(p: Path) -> dict:
    now = time.time()
    try:
        if not p.exists():
            return {"exists": False, "age_s": None, "mtime": None}
        st = p.stat()
        return {
            "exists": True,
            "age_s": round(now - float(st.st_mtime), 3),
            "mtime": float(st.st_mtime),
        }
    except Exception as e:
        return {"exists": False, "age_s": None, "mtime": None, "error": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.set_main_loop(asyncio.get_running_loop())
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
                ".\\main.py:124",
                "Suppressed exception (was 'pass')",
            )

    hb_stop = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat_worker, args=(hb_stop,), daemon=True)
    hb_thread.start()
    lb_stop = asyncio.Event()
    lb_task = asyncio.create_task(_loopbeat_worker(lb_stop))

    hm_stop = threading.Event()
    hm_thread = None
    if HANG_MONITOR_ENABLE:
        hm_thread = threading.Thread(
            target=_hang_monitor_worker, args=(hm_stop,), daemon=True
        )
        hm_thread.start()

    if CFG.get("pipeline_enabled", False):
        LOG.info("Startup: Initializing pipeline.")
        try:
            ensure_pipeline_loaded()
        except Exception as e:
            LOG.error(f"Startup Warning: Could not load pipeline: {e}")
    else:
        LOG.info("Startup: Pipeline disabled via config.")

    loop_task = asyncio.create_task(gui_api.broadcast_status_loop())

    yield

    LOG.info("Shutdown: Stopping Broadcast Loop...")
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    except Exception:
        import logging as _logging

        try:
            from app.utils import log_exception_throttled
        except Exception:
            _logging.getLogger(__name__).exception(
                "Shutdown: loop_task failed [throttle helper import failed]"
            )
        else:
            log_exception_throttled(
                _logging.getLogger(__name__),
                r".\main.py:shutdown_loop_task",
                "Shutdown: loop_task failed",
            )

    lb_stop.set()
    lb_task.cancel()
    try:
        await lb_task
    except asyncio.CancelledError:
        pass
    except Exception:
        import logging as _logging

        try:
            from app.utils import log_exception_throttled
        except Exception:
            _logging.getLogger(__name__).exception(
                "Shutdown: lb_task failed [throttle helper import failed]"
            )
        else:
            log_exception_throttled(
                _logging.getLogger(__name__),
                r".\main.py:shutdown_lb_task",
                "Shutdown: lb_task failed",
            )

    try:
        if HANG_MONITOR_ENABLE:
            hm_stop.set()
            if hm_thread:
                hm_thread.join(timeout=1.0)
    except Exception:
        pass

    hb_stop.set()

    LOG.info("Shutdown: Cleaning up resources...")
    from app.state import stop_pipeline

    try:
        stop_pipeline()
    except Exception:
        import logging as _logging

        try:
            from app.utils import log_exception_throttled
        except Exception:
            _logging.getLogger(__name__).exception(
                "Shutdown: cleanup failed [throttle helper import failed]"
            )
        else:
            log_exception_throttled(
                _logging.getLogger(__name__),
                r".\main.py:shutdown_cleanup",
                "Shutdown: cleanup failed",
            )

    LOG.info("Shutdown: Cleanup complete.")


app = FastAPI(title="OBF Vision API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(vision.router)
app.include_router(gui_api.router)


@app.get("/gui", response_class=HTMLResponse)
async def gui_page():
    tpl = Path("app/templates/gui.html")
    if tpl.exists():
        return tpl.read_text(encoding="utf-8")
    return "GUI Template not found (check app/templates/gui.html)."


@app.get("/")
async def root():
    return "OBF Vision Service Ready"


@app.get("/health")
async def health():
    hb = _file_age_info(HEARTBEAT_FILE)
    lb = _file_age_info(LOOPBEAT_FILE)

    try:
        with state.STATE_LOCK:
            pipe_enabled = bool(state.PIPE_ENABLED)
        perf = state.PERF
        perf_info = {
            "req_count": int(getattr(perf, "req_count", 0) or 0),
            "ema_ms": float(getattr(perf, "ema_ms", 0.0) or 0.0),
            "last_ms": float(getattr(perf, "last_ms", 0.0) or 0.0),
        }
    except Exception:
        pipe_enabled = bool(state.PIPE_ENABLED)
        perf_info = {}

    return {
        "ok": True,
        "pid": os.getpid(),
        "uptime_s": int(time.time() - state.START_TIME),
        "pipeline_enabled": pipe_enabled,
        "heartbeat": hb,
        "loopbeat": lb,
        **({"perf": perf_info} if perf_info else {}),
    }


def main():
    try:
        host = str(CFG.get("host", "0.0.0.0"))
        port = int(CFG.get("port", 32168))

        LOG.info(f"Starting server on {host}:{port}")

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_config=None,
            log_level=str(LOG_LEVEL_STR).lower(),
            access_log=False,
            reload=False,
            timeout_keep_alive=65,
            limit_concurrency=100,
        )
    except Exception as e:
        LOG.critical(f"FATAL ERROR in main loop: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
