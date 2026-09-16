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
import warnings
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
    "stale_cluster_wheels",
    "upload_bytes",
    "upload_wheel",
    "uploaded_wheel",
    "warn_stale_cluster_wheels",
    "wheel_exists",
]

# Everything that can change what the job does.  Docs and tests are excluded
# deliberately: they cannot alter job behaviour and would cause needless
# rebuilds during development.
HASHED_PATHS: tuple[str, ...] = ("src", "pyproject.toml")

_MEMO: dict[tuple[str, str], str] = {}

#: Clusters already reported on. The check costs an API call and the answer
#: cannot change without a restart, so once per process per cluster is enough.
_WARNED_CLUSTERS: set[str] = set()


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


def stale_cluster_wheels(client: Any, cluster_id: str, keep: str = "") -> list[str]:
    """``amt-*`` wheels installed on ``cluster_id`` other than ``keep``.

    Cluster-scoped installs accumulate: every run against an existing cluster
    adds one, and pip keeps serving whichever it resolved first among installs
    that look identical to it.
    """
    try:
        statuses = list(client.libraries.cluster_status(cluster_id=cluster_id))
    except Exception:  # diagnostics only; a lost lookup must not fail the run
        return []
    stale = []
    for status in statuses:
        whl = str(getattr(getattr(status, "library", None), "whl", "") or "")
        if whl and whl != keep and Path(whl).name.startswith("amt-"):
            stale.append(whl)
    return stale


def warn_stale_cluster_wheels(client: Any, cluster_id: str, keep: str = "") -> list[str]:
    """Warn about older ``amt`` installs on an all-purpose cluster.

    This warns rather than uninstalling. A library uninstall on an all-purpose
    cluster only takes effect after a restart, and restarting a cluster out
    from under a campaign -- possibly one with another run on it -- is not a
    decision a wheel-upload helper gets to make. ``scripts/clear_cluster_wheels.py``
    is the deliberate, operator-run version of the same thing.
    """
    if not cluster_id or cluster_id in _WARNED_CLUSTERS:
        return []
    _WARNED_CLUSTERS.add(cluster_id)
    stale = stale_cluster_wheels(client, cluster_id, keep)
    if stale:
        names = ", ".join(sorted(Path(p).name for p in stale))
        warnings.warn(
            f"cluster {cluster_id} carries {len(stale)} other amt wheel "
            f"install(s): {names}. The build stamp in amt._version normally "
            f"makes pip prefer {Path(keep).name or 'the new wheel'}, but an "
            "install predating that stamp can still shadow it, and the job "
            "then fails on arguments this code clearly has. Clearing them "
            "needs a cluster restart, which this process will not trigger: run "
            f"`uv run python scripts/clear_cluster_wheels.py {cluster_id}`.",
            stacklevel=2,
        )
    return stale


def ensure_wheel(
    client: Any,
    volume_path: str,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
    cluster_id: str | None = None,
) -> str:
    """Return a Volume path holding a wheel built from the current source.

    Builds and uploads only when the source-tree hash is not already present,
    so repeated trainer iterations reuse one artifact. ``cluster_id``, when the
    runs go to an existing cluster, is checked for stale installs of this same
    package -- see :func:`warn_stale_cluster_wheels`.
    """
    remote = _resolve_wheel(client, volume_path, root=root, force=force)
    if cluster_id:
        warn_stale_cluster_wheels(client, str(cluster_id), remote)
    return remote


def _resolve_wheel(
    client: Any,
    volume_path: str,
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> str:
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
