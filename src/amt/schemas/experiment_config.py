"""Stage 4 — ExperimentConfig (SPEC.md §Stage 4). Configuration-driven training."""

from __future__ import annotations

from pydantic import Field

from amt.schemas.common import AmtModel, new_id, stable_hash
from amt.schemas.feature_recipe import FeatureRecipe, FitSpec


class ExperimentConfig(AmtModel):
    """A single, fully-specified training configuration.

    ``model``/``training``/``loss``/``features`` mirror the YAML shape in SPEC.md §Stage 4.
    """

    config_id: str = Field(default_factory=lambda: new_id("cfg"))
    hypothesis_id: str = ""
    family: str = Field(description="Model family, e.g. 'sasrec' or 'popularity'.")
    model: dict = Field(default_factory=dict)
    training: dict = Field(default_factory=dict)
    loss: dict = Field(default_factory=dict)
    features: dict = Field(default_factory=dict)
    # Tabular families only: the approved input shape, stamped in by the graph.
    feature_recipe: FeatureRecipe | None = None
    fit_spec: FitSpec | None = None
    feature_fit_hash: str = Field(
        default="", description="Approved feature/fit identity this config runs under."
    )
    rationale: str = Field(default="", description="Why each significant change was made.")
    changes_from_previous: list[str] = Field(default_factory=list)
    custom_code_ref: str | None = Field(
        default=None, description="Only when the hypothesis genuinely requires custom code."
    )
    contract_hash: str = ""

    @property
    def config_hash(self) -> str:
        """Identity of the trained artifact's configuration (excludes prose and ids)."""
        return stable_hash(
            {
                "family": self.family,
                "model": self.model,
                "training": self.training,
                "loss": self.loss,
                "features": self.features,
                "custom_code_ref": self.custom_code_ref,
                "feature_fit_hash": self.feature_fit_hash,
                # Hyperparameters live here for the boosters, so the full spec counts.
                "fit_spec": self.fit_spec.model_dump(mode="json") if self.fit_spec else None,
            }
        )

    def to_yaml_dict(self) -> dict:
        out: dict = {
            "model": {"family": self.family, **self.model},
            "training": self.training,
            "loss": self.loss,
            "features": self.features,
        }
        if self.feature_recipe is not None:
            out["feature_recipe"] = self.feature_recipe.model_dump(mode="json")
        if self.fit_spec is not None:
            out["fit_spec"] = self.fit_spec.model_dump(mode="json")
        return out

    def flat_params(self) -> dict[str, object]:
        """Flattened params for MLflow logging."""
        out: dict[str, object] = {"family": self.family}
        for section in ("model", "training", "loss", "features"):
            for k, v in getattr(self, section).items():
                out[f"{section}.{k}"] = v
        if self.fit_spec is not None:
            for k, v in self.fit_spec.model_dump(mode="json").items():
                if k == "booster_params":
                    continue
                out[f"fit.{k}"] = v
            for k, v in self.fit_spec.booster_params.items():
                out[f"booster.{k}"] = v
        if self.feature_recipe is not None:
            out["feature_recipe.n_features"] = len(self.feature_recipe.feature_columns)
            out["feature_recipe.label_strategy"] = self.feature_recipe.label.strategy
            out["feature_recipe.candidates"] = self.feature_recipe.candidate_generation.strategy
        if self.feature_fit_hash:
            out["feature_fit_hash"] = self.feature_fit_hash
        return out
