"""The package version, which the build stamps to make wheels distinguishable.

Databricks installs a job's wheel as a cluster-scoped library with
``pip install --upgrade``, and pip skips a wheel whose version is already
installed no matter which path it came from.  Two builds of this repo that both
call themselves ``0.1.0`` are therefore indistinguishable to the cluster: the
first one installed keeps serving every later run.  ``AMT_WHEEL_LOCAL_VERSION``
carries a build stamp into the version so each build is an upgrade.

Read by hatchling at build time (see ``[tool.hatch.version]``), so this module
must stay importable on its own.
"""

from __future__ import annotations

import os
import re

BASE_VERSION = "0.1.0"

_LOCAL_VERSION = re.compile(r"^[a-z0-9]+(\.[a-z0-9]+)*$")


def version() -> str:
    """``BASE_VERSION``, plus the local version segment the build asked for."""
    stamp = os.environ.get("AMT_WHEEL_LOCAL_VERSION", "").strip()
    if not stamp:
        return BASE_VERSION
    if not _LOCAL_VERSION.match(stamp):
        raise ValueError(
            f"AMT_WHEEL_LOCAL_VERSION={stamp!r} is not a PEP 440 local version: "
            "lowercase alphanumeric segments separated by periods"
        )
    return f"{BASE_VERSION}+{stamp}"
