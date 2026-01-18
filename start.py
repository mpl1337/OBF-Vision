import atexit
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from app.depcheck import assert_runtime_deps

    assert_runtime_deps(extra_imports=["pydantic"])
except SystemExit:
    raise
except Exception:
    import traceback

    print("[WATCHDOG] Dependency-Check fehlgeschlagen:", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)

SCRIPT_NAME = "main.py"


LOOP_SLEEP = 2.0

HEARTBEAT_FILE = Path(".obf_heartbeat")
LOOPBEAT_FILE = Path(".obf_loopbeat")

STALE_AFTER_SEC = 15
LOOPBEAT_STALE_AFTER_SEC = 10
STARTUP_GRACE_SEC = 20


class Watchdog:
    def __init__(self):
        self.process = None
        self.running = True
        self.last_start_ts = 0.0

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts} WATCHDOG] {msg}", flush=True)

    def _spawn_kwargs(self):
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        return kw

    def _dump_traces_before_kill(self):
        if not getattr(self, "process", None):
            return
        try:
            if sys.platform == "win32":
                self.log("Sende CTRL+BREAK (Stackdump) ...")
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(1.0)
        except Exception as e:
            self.log(f"Stackdump senden fehlgeschlagen: {e}")

    def start_server(self):
        self.log(f"Starte {SCRIPT_NAME}...")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        for f in [HEARTBEAT_FILE, LOOPBEAT_FILE]:
            try:
                if f.exists():
                    f.unlink()
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
                        ".\\start.py:61",
                        "Suppressed exception (was 'pass')",
                    )

        try:
            self.process = subprocess.Popen(
                [sys.executable, SCRIPT_NAME],
                env=env,
                cwd=os.getcwd(),
                **self._spawn_kwargs(),
            )
            self.last_start_ts = time.time()
            time.sleep(3)
        except Exception as e:
            self.log(f"FEHLER beim Starten: {e}")
            self.running = False

    def _terminate_hard(self):
        if not self.process:
            return
        try:
            self.log("Versuche terminate() ...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
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
                        ".\\start.py:86",
                        "Suppressed exception (was 'pass')",
                    )

            self.log("terminate() hat nicht gereicht -> kill() ...")
            self.process.kill()
            try:
                self.process.wait(timeout=5)
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
                        ".\\start.py:93",
                        "Suppressed exception (was 'pass')",
                    )
        except Exception as e:
            self.log(f"Fehler beim Beenden: {e}")

    def _heartbeat_stale(self) -> bool:
        now = time.time()

        if (now - self.last_start_ts) < STARTUP_GRACE_SEC:
            return False

        try:
            if not HEARTBEAT_FILE.exists():
                return True
            mtime = HEARTBEAT_FILE.stat().st_mtime
            return (now - mtime) > STALE_AFTER_SEC
        except Exception:
            return True

    def _loopbeat_stale(self) -> bool:
        now = time.time()

        if (now - self.last_start_ts) < STARTUP_GRACE_SEC:
            return False

        try:
            if not LOOPBEAT_FILE.exists():
                return True
            mtime = LOOPBEAT_FILE.stat().st_mtime
            return (now - mtime) > LOOPBEAT_STALE_AFTER_SEC
        except Exception:
            return True

    def _process_dead(self) -> bool:
        if not self.process:
            return True
        return self.process.poll() is not None

    def stop(self, reason: str = "stop"):
        self.log(f"Stop requested: {reason}")
        self.running = False
        try:
            if self.process and self.process.poll() is None:
                self._dump_traces_before_kill()
                self._terminate_hard()
        except Exception as e:
            self.log(f"Stop error: {e}")

    def _sig_handler(self, sig, frame):
        self.stop(f"signal {sig}")

    def run(self):
        self.start_server()
        try:
            while self.running:
                time.sleep(LOOP_SLEEP)

                if not self.running:
                    break

                if self.process is None:
                    if not self.running:
                        break
                    self.log("Prozess ist None. Neustart...")
                    self.start_server()
                    continue

                if self._process_dead():
                    if not self.running:
                        break
                    self.log("Prozess beendet. Neustart...")
                    self.start_server()
                    continue

                if self._heartbeat_stale():
                    if not self.running:
                        break
                    self.log(f"HEARTBEAT stale > {STALE_AFTER_SEC}s. Neustart...")
                    self._dump_traces_before_kill()
                    self._terminate_hard()
                    if not self.running:
                        break
                    self.start_server()
                    continue

                if self._loopbeat_stale():
                    if not self.running:
                        break
                    self.log(
                        f"LOOPBEAT stale > {LOOPBEAT_STALE_AFTER_SEC}s. Neustart..."
                    )
                    self._dump_traces_before_kill()
                    self._terminate_hard()
                    if not self.running:
                        break
                    self.start_server()
                    continue

        except KeyboardInterrupt:
            self.stop("KeyboardInterrupt (Ctrl+C)")
            return
        finally:
            if not self.running:
                self.stop("finally cleanup")


if __name__ == "__main__":
    wd = Watchdog()

    def _sig_handler(sig, frame):
        wd.stop(f"signal {sig}")
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    atexit.register(lambda: wd.stop("atexit"))
    wd.run()
