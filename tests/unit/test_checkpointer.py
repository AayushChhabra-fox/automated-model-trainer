"""What survives a round trip through the SQLite checkpointer.

The serializer's allowlist is the difference between a resumed campaign holding
``EvaluationContract`` objects and holding dicts that look like them, so it is
worth a test rather than a warning nobody reads.
"""

from __future__ import annotations

import warnings

from amt.graph.build import CHECKPOINT_PACKAGES, checkpoint_types, sqlite_checkpointer
from amt.schemas.common import AmtModel
from amt.settings import Settings


def _serde():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types())


def test_the_campaign_schemas_come_back_as_themselves(caplog, evaluation_contract, frozen_champion):
    serde = _serde()
    for value in (evaluation_contract, frozen_champion, frozen_champion.config):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            restored = serde.loads_typed(serde.dumps_typed(value))
        assert type(restored) is type(value)
        assert restored == value
    # The seventeen lines in front of every command.
    assert "unregistered type" not in caplog.text


def test_a_type_outside_the_allowlist_comes_back_as_a_dict(caplog):
    """The failure mode `checkpoint_types` is enumerated to avoid."""

    class Stranger(AmtModel):
        name: str = "x"

    serde = _serde()
    restored = serde.loads_typed(serde.dumps_typed(Stranger()))

    assert isinstance(restored, dict)
    assert "Blocked deserialization" in caplog.text


def test_every_schema_in_the_state_bearing_packages_is_allowed():
    """A schema added later must not have to be remembered here."""
    allowed = set(checkpoint_types())
    missing = {
        cls
        for cls in _all_subclasses(AmtModel)
        if cls.__module__.startswith(CHECKPOINT_PACKAGES) and cls not in allowed
    }
    assert not missing


def _all_subclasses(base: type) -> set[type]:
    import amt.schemas  # noqa: F401  (registers every schema as a subclass)

    found = set()
    for cls in base.__subclasses__():
        found.add(cls)
        found |= _all_subclasses(cls)
    return found


def test_a_reader_waits_for_the_writer_instead_of_failing_on_the_lock(tmp_path):
    """`amt status` opens this file while a runner is writing it."""
    settings = Settings(_env_file=None, AMT_CHECKPOINT_DB=str(tmp_path / "checkpoints.sqlite"))

    saver = sqlite_checkpointer(settings=settings)

    assert saver.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
