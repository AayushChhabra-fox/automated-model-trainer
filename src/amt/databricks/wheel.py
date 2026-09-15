"""Build the ``amt`` wheel and upload it to a UC Volume.

Databricks runs *this* repo's code, so the wheel has to be rebuilt whenever
the source changes -- but rebuilding and re-uploading on every job submission
would add a minute to every trainer iteration.  The compromise is a
content-addressed cache: the wheel is named after a hash of the source tree,
so an unchanged tree resolves to an already-uploaded artifact and the build
is skipped entirely.

One consequence of the cache: a cached wheel keeps the build stamp it was first
built with, so going *back* to a tree that was built earlier than the one a
long-lived cluster already has installed needs ``force=True`` to count as an
upgrade.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "HASHED_PATHS",
    "WheelError",
    "build_stamp",
    "build_wheel",
    "ensure_wheel",
    "remote_wheel_dir",
    "remote_wheel_path",
    "repo_root",
    "source_tree_hash",
    "upload_bytes",
    "upload_wheel",
    "uploaded_wheel",
    "wheel_exists",
]

# Everything that can change what the job does.  Docs and tests are excluded
# deliberately: they cannot alter job behaviour and would cause needless
# rebuilds during development.
HASHED_PATHS: tuple[str, ...] = ("src", "pyproject.toml")

_MEMO: dict[tuple[str, str], str] = {}


class WheelError(RuntimeError):
    """Raised when the wheel cannot be built or uploaded."""


def repo_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Walk up from ``start`` until a ``pyproject.toml`` is found."""
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise WheelError(f"no pyproject.toml above {here}")


def _iter_files(root: Path, paths: Iterable[str]):
    for rel in paths:
        target = root / rel
        if target.is_file():
            yield target
        elif target.is_dir():
            for path in sorted(target.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    yield path


def source_tree_hash(root: str | os.PathLike[str] | None = None) -> str:
    """sha256 over the relative path and bytes of every source file."""
    base = Path(root) if root else repo_root()
    digest = hashlib.sha256()
    for path in sorted(_iter_files(base, HASHED_PATHS)):
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_stamp(tree_hash: str, now: float | None = None) -> str:
    """A PEP 440 local version that increases with every build.

    The timestamp leads so pip compares builds numerically and always treats a
    newer one as an upgrade; the hash makes the version name the source it came
    from.
    """
    moment = time.gmtime(now) if now is not None else time.gmtime()
    return f"{time.strftime('%Y%m%d%H%M%S', moment)}.{tree_hash[:12]}"


def build_wheel(
    root: str | os.PathLike[str] | None = None,
    out_dir: str | None = None,
    local_version: str = "",
) -> Path:
    """Build a wheel with ``uv build`` (falling back to ``python -m build``)."""
    base = Path(root) if root else repo_root()
    dest = Path(out_dir or tempfile.mkdtemp(prefix="amt-wheel-"))
    dest.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if local_version:
        env["AMT_WHEEL_LOCAL_VERSION"] = local_version

    commands = []
    if shutil.which("uv"):
        commands.append(["uv", "build", "--wheel", "--out-dir", str(dest)])
    commands.append(
        ["python", "-m", "build", "--wheel", "--outdir", str(dest), str(base)]
    )

    last: Exception | None = None
    for cmd in commands:
        try:
            subprocess.run(cmd, cwd=str(base), check=True, capture_output=True, env=env)
        except Exception as exc:
            last = exc
            continue
        wheels = sorted(dest.glob("*.whl"), key=lambda p: p.stat().st_mtime)
        if wheels:
            return wheels[-1]
    raise WheelError(f"wheel build failed in {base}: {last}")


def remote_wheel_dir(volume_path: str, tree_hash: str) -> str:
    """One directory per source-tree hash."""
    return f"{volume_path.rstrip('/')}/wheels/{tree_hash[:12]}"


def remote_wheel_path(volume_path: str, tree_hash: str, wheel_name: str) -> str:
    """Content-addressed destination under the artifact Volume.

    The hash is a directory rather than a filename suffix: pip parses wheel names
    per PEP 427, and an extra segment after the platform tag is read as a build tag,
    which must start with a digit. ``amt-0.1.0-py3-none-any-<hash>.whl`` is rejected
    outright with "is not a valid wheel filename".
    """
    return f"{remote_wheel_dir(volume_path, tree_hash)}/{Path(wheel_name).name}"


def upload_bytes(client: Any, remote_path: str, payload: bytes) -> str:
    """Upload raw bytes to a UC Volume path via the Files API."""
    import io

    try:
        client.files.upload(remote_path, io.BytesIO(payload), overwrite=True)
    except Exception as exc:
        raise WheelError(f"upload to {remote_path} failed: {exc}") from exc
    return remote_path


def upload_wheel(client: Any, local_path: str | os.PathLike[str], remote_path: str) -> str:
    return upload_bytes(client, remote_path, Path(local_path).read_bytes())


def wheel_exists(client: Any, remote_path: str) -> bool:
    try:
        client.files.get_metadata(remote_path)
        return True
    except Exception:
        return False


def uploaded_wheel(client: Any, directory: str) -> str | None:
    """The wheel already sitting in ``directory``, if any."""
    try:
        entries = list(client.files.list_directory_contents(directory))
    except Exception:  # the directory simply does not exist yet
        return None
    for entry in entries:
        path = str(getattr(entry, "path", "") or "")
        if path.endswith(".whl"):
            return path
    return None


def ensure_wheel(
    client: Any,
    volume_path: str,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> str:
    """Return a Volume path holding a wheel built from the current source.

    Builds and uploads only when the source-tree hash is not already present,
    so repeated trainer iterations reuse one artifact.
    """
    base = Path(root) if root else repo_root()
    tree_hash = source_tree_hash(base)
    memo_key = (str(base), tree_hash)
    if not force and memo_key in _MEMO:
        return _MEMO[memo_key]

    # Look in the hash's directory first, so a fresh process can skip the build
    # without having to guess what the built wheel will be called.
    if not force:
        existing = uploaded_wheel(client, remote_wheel_dir(volume_path, tree_hash))
        if existing:
            _MEMO[memo_key] = existing
            return existing

    with tempfile.TemporaryDirectory(prefix="amt-wheel-") as tmp:
        local = build_wheel(base, out_dir=tmp, local_version=build_stamp(tree_hash))
        remote = remote_wheel_path(volume_path, tree_hash, local.name)
        if force or not wheel_exists(client, remote):
            upload_wheel(client, local, remote)
    _MEMO[memo_key] = remote
    return remote
