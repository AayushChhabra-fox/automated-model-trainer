"""Wheel naming, versioning, and the content-addressed upload cache."""

from __future__ import annotations

from typing import Any

import pytest
from packaging.utils import parse_wheel_filename
from packaging.version import Version

from amt._version import BASE_VERSION, version
from amt.databricks.wheel import (
    build_stamp,
    remote_wheel_dir,
    remote_wheel_path,
    uploaded_wheel,
)

VOLUME = "/Volumes/cat/schema/amt_artifacts"
TREE_HASH = "65a806a9e8071234"
WHEEL = "amt-0.1.0-py3-none-any.whl"


def test_the_uploaded_name_stays_a_valid_wheel_filename() -> None:
    """pip reads a sixth segment as a build tag and rejects the whole file."""
    remote = remote_wheel_path(VOLUME, TREE_HASH, WHEEL)

    assert remote.rsplit("/", 1)[1] == WHEEL
    assert len(WHEEL.removesuffix(".whl").split("-")) == 5


def test_the_hash_addresses_a_directory() -> None:
    remote = remote_wheel_path(VOLUME, TREE_HASH, WHEEL)

    assert remote == f"{VOLUME}/wheels/{TREE_HASH[:12]}/{WHEEL}"
    assert remote_wheel_dir(VOLUME, TREE_HASH).endswith(TREE_HASH[:12])


class _Entry:
    def __init__(self, path: str) -> None:
        self.path = path


class _Files:
    def __init__(self, listing: dict[str, list[str]]) -> None:
        self.listing = listing

    def list_directory_contents(self, directory: str) -> list[_Entry]:
        if directory not in self.listing:
            raise FileNotFoundError(directory)
        return [_Entry(p) for p in self.listing[directory]]


class _Client:
    def __init__(self, listing: dict[str, list[str]]) -> None:
        self.files: Any = _Files(listing)


def test_the_cache_probe_finds_a_wheel_without_guessing_its_name() -> None:
    directory = remote_wheel_dir(VOLUME, TREE_HASH)
    client = _Client({directory: [f"{directory}/marker.txt", f"{directory}/{WHEEL}"]})

    assert uploaded_wheel(client, directory) == f"{directory}/{WHEEL}"


def test_a_missing_directory_is_a_cache_miss() -> None:
    assert uploaded_wheel(_Client({}), remote_wheel_dir(VOLUME, TREE_HASH)) is None


def test_every_build_is_an_upgrade_over_the_last_one() -> None:
    """pip skips a wheel whose version is installed, whatever path it came from."""
    first = Version(f"{BASE_VERSION}+{build_stamp(TREE_HASH, now=1_600_000_000)}")
    second = Version(f"{BASE_VERSION}+{build_stamp('0123456789ab', now=1_600_000_060)}")

    assert Version(BASE_VERSION) < first < second


def test_a_stamped_wheel_keeps_a_name_pip_can_read(monkeypatch: Any) -> None:
    monkeypatch.setenv("AMT_WHEEL_LOCAL_VERSION", build_stamp(TREE_HASH))
    stamped = version()

    name, parsed, _build, _tags = parse_wheel_filename(f"amt-{stamped}-py3-none-any.whl")

    assert (name, parsed) == ("amt", Version(stamped))


def test_an_unstamped_build_is_the_plain_version(monkeypatch: Any) -> None:
    monkeypatch.delenv("AMT_WHEEL_LOCAL_VERSION", raising=False)

    assert version() == BASE_VERSION


def test_a_stamp_pep_440_cannot_express_is_refused(monkeypatch: Any) -> None:
    monkeypatch.setenv("AMT_WHEEL_LOCAL_VERSION", "2026-09-14+dirty")

    with pytest.raises(ValueError, match="local version"):
        version()
