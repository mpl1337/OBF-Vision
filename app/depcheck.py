from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from importlib import metadata
except Exception:
    metadata = None


PIP_IMPORT_MAP: dict[str, str] = {
    "faiss-cpu": "faiss",
    "fastapi": "fastapi",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "openvino": "openvino",
    "pillow": "PIL",
    "python-multipart": "multipart",
    "ultralytics": "ultralytics",
    "uvicorn": "uvicorn",
    "websockets": "websockets",
    "pydantic": "pydantic",
}


DISPLAY_NAME: dict[str, str] = {
    "PIL": "Pillow (PIL)",
}


@dataclass(frozen=True)
class DepStatus:
    dist: str
    import_name: str
    pinned: str | None
    installed: str | None
    ok: bool
    error: str | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_requirements(req_path: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if not req_path.exists():
        return out

    for raw in req_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("-"):
            continue

        name = line
        ver: str | None = None

        if "==" in line:
            parts = line.split("==", 1)
            name = parts[0].strip()
            ver = parts[1].strip() or None
        elif ">=" in line:
            parts = line.split(">=", 1)
            name = parts[0].strip()
            ver = None
        elif "<=" in line or "~=" in line or ">" in line or "<" in line:
            name = re.split(r"[<>~=]", line, maxsplit=1)[0].strip()
            ver = None

        if not name:
            continue

        out[name.lower()] = ver
    return out


def _safe_dist_version(dist: str) -> str | None:
    if metadata is None:
        return None
    try:
        return metadata.version(dist)
    except Exception:
        return None


def _import_check(import_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(import_name)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def assert_runtime_deps(
    requirements_path: Path | None = None,
    *,
    strict_versions: bool = False,
    extra_imports: list[str] | None = None,
) -> list[DepStatus]:
    root = _project_root()
    req_path = requirements_path or (root / "requirements.txt")

    pinned = _parse_requirements(req_path)

    deps: list[tuple[str, str, str | None]] = []

    for dist_lc, ver in sorted(pinned.items(), key=lambda x: x[0]):
        imp = PIP_IMPORT_MAP.get(dist_lc, dist_lc.replace("-", "_"))
        deps.append((dist_lc, imp, ver))

    for imp in extra_imports or []:
        deps.append((imp.lower(), imp, None))

    seen = set()
    uniq: list[tuple[str, str, str | None]] = []
    for dist, imp, ver in deps:
        if imp in seen:
            continue
        seen.add(imp)
        uniq.append((dist, imp, ver))
    deps = uniq

    results: list[DepStatus] = []
    missing: list[DepStatus] = []
    mismatched: list[DepStatus] = []

    for dist, imp, ver in deps:
        ok, err = _import_check(imp)
        installed = _safe_dist_version(dist)
        st = DepStatus(
            dist=dist,
            import_name=imp,
            pinned=ver,
            installed=installed,
            ok=ok,
            error=err,
        )
        results.append(st)
        if not ok:
            missing.append(st)
        elif strict_versions and ver and installed and (installed != ver):
            mismatched.append(st)

    if missing or mismatched:
        print("\n[DEPS] Dependency preflight failed.", file=sys.stderr)
        if req_path.exists():
            print(f"[DEPS] Checked: {req_path}", file=sys.stderr)
        print("[DEPS]", file=sys.stderr)

        if missing:
            print("[DEPS] Missing/broken imports:", file=sys.stderr)
            for st in missing:
                disp = DISPLAY_NAME.get(st.import_name, st.import_name)
                pin = f"=={st.pinned}" if st.pinned else ""
                print(
                    f"  - {st.dist}{pin}  (import '{disp}') -> {st.error}",
                    file=sys.stderr,
                )

        if mismatched:
            print("[DEPS] Version mismatches (strict_versions=True):", file=sys.stderr)
            for st in mismatched:
                print(
                    f"  - {st.dist} expected {st.pinned}, installed {st.installed}",
                    file=sys.stderr,
                )

        print("\n[DEPS] Fix:", file=sys.stderr)
        print("  python -m pip install -r requirements.txt", file=sys.stderr)
        print("", file=sys.stderr)

        raise SystemExit(2)

    return results
