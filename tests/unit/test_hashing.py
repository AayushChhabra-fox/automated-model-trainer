"""Hash stability — the backbone of the campaign's ground-truth enforcement."""

from __future__ import annotations

from amt.schemas.common import stable_hash


def test_stable_hash_is_key_order_independent():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})


def test_stable_hash_changes_with_content():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})


def test_spec_hash_ignores_timestamps_and_ids(problem_spec):
    other = problem_spec.model_copy(
        update={"spec_id": "spec-other", "created_at": problem_spec.created_at.replace(year=2030)}
    )
    assert other.spec_hash == problem_spec.spec_hash


def test_spec_hash_tracks_meaningful_change(problem_spec):
    changed = problem_spec.model_copy(update={"primary_offline_metric": "recall@10"})
    assert changed.spec_hash != problem_spec.spec_hash


def test_contract_hash_is_stable_across_copies(evaluation_contract):
    copy = evaluation_contract.model_copy(deep=True)
    assert copy.contract_hash == evaluation_contract.contract_hash
    reloaded = evaluation_contract.model_validate(evaluation_contract.model_dump())
    assert reloaded.contract_hash == evaluation_contract.contract_hash


def test_contract_hash_changes_when_ground_truth_changes(evaluation_contract):
    moved_split = evaluation_contract.model_copy(
        update={
            "qa_split": evaluation_contract.qa_split.model_copy(
                update={"membership_hash": "different"}
            )
        }
    )
    assert moved_split.contract_hash != evaluation_contract.contract_hash

    relabeled = evaluation_contract.model_copy(update={"label_definition": "something else"})
    assert relabeled.contract_hash != evaluation_contract.contract_hash


def test_config_hash_ignores_prose_but_tracks_parameters(experiment_config):
    same = experiment_config.model_copy(
        update={"rationale": "completely different words", "config_id": "cfg-other"}
    )
    assert same.config_hash == experiment_config.config_hash

    different = experiment_config.model_copy(
        update={"training": {**experiment_config.training, "learning_rate": 0.1}}
    )
    assert different.config_hash != experiment_config.config_hash


def test_config_flattening_for_mlflow(experiment_config):
    params = experiment_config.flat_params()
    assert params["family"] == experiment_config.family
    assert any(key.startswith("training.") for key in params)
    assert experiment_config.to_yaml_dict()["model"]["family"] == experiment_config.family
