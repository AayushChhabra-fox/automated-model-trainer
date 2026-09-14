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


def _recipe(**overrides):
    from amt.schemas.feature_recipe import (
        CandidateGeneration,
        FeatureRecipe,
        FeatureTransform,
        LabelSpec,
        RecipeSource,
    )

    fields = {
        "sources": [RecipeSource(name="interactions", role="interactions", split="train")],
        "feature_columns": [
            FeatureTransform(name="pop", kind="count", of="item_id", over="train"),
            FeatureTransform(name="hist_len", kind="column", source="history_length"),
        ],
        "label": LabelSpec(strategy="holdout_next_item"),
        "candidate_generation": CandidateGeneration(strategy="popularity_topn", n=200),
    }
    fields.update(overrides)
    return FeatureRecipe(**fields)


def test_feature_fit_hash_ignores_booster_params():
    from amt.schemas.feature_recipe import FitSpec, feature_fit_hash

    recipe = _recipe()
    base = FitSpec(num_boost_round=200, booster_params={"learning_rate": 0.1})
    retuned = FitSpec(num_boost_round=200, booster_params={"learning_rate": 0.01, "max_depth": 9})
    assert feature_fit_hash(recipe, base) == feature_fit_hash(recipe, retuned)


def test_feature_fit_hash_tracks_structural_changes():
    from amt.schemas.feature_recipe import FeatureTransform, FitSpec, feature_fit_hash

    recipe = _recipe()
    fit = FitSpec()
    assert feature_fit_hash(recipe, fit) != feature_fit_hash(
        recipe, fit.model_copy(update={"num_boost_round": 500})
    )
    wider = _recipe(
        feature_columns=[
            *recipe.feature_columns,
            FeatureTransform(name="bias", kind="literal", value=1.0),
        ]
    )
    assert feature_fit_hash(wider, fit) != feature_fit_hash(recipe, fit)
    relabelled = recipe.model_copy(
        update={"label": recipe.label.model_copy(update={"strategy": "interaction_positive"})}
    )
    assert feature_fit_hash(relabelled, fit) != feature_fit_hash(recipe, fit)


def test_feature_fit_hash_ignores_ids_and_prose():
    from amt.schemas.feature_recipe import FitSpec, feature_fit_hash

    recipe = _recipe()
    reworded = recipe.model_copy(
        update={
            "recipe_id": "recipe-other",
            "notes": "rewritten explanation",
            "leakage_notes": "same argument, new words",
        }
    )
    assert feature_fit_hash(reworded, FitSpec()) == feature_fit_hash(recipe, FitSpec())


def test_config_hash_tracks_the_feature_fit_shape(experiment_config):
    from amt.schemas.feature_recipe import FitSpec, feature_fit_hash

    recipe = _recipe()
    tabular = experiment_config.model_copy(
        update={
            "family": "lightgbm_reranker",
            "feature_recipe": recipe,
            "fit_spec": FitSpec(booster_params={"learning_rate": 0.1}),
            "feature_fit_hash": feature_fit_hash(recipe, FitSpec()),
        }
    )
    assert tabular.config_hash != experiment_config.config_hash
    # Hyperparameters are part of the run's identity even though approval ignores them.
    retuned = tabular.model_copy(
        update={"fit_spec": tabular.fit_spec.model_copy(update={"booster_params": {"eta": 0.3}})}
    )
    assert retuned.config_hash != tabular.config_hash
    assert retuned.feature_fit_hash == tabular.feature_fit_hash
