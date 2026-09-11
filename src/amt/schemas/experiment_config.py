"""Stage 4 — ExperimentConfig (SPEC.md §Stage 4). Configuration-driven training."""

from __future__ import annotations

from pydantic import Field

from amt.schemas.common import AmtModel, new_id, stable_hash


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
            }
        )

    def to_yaml_dict(self) -> dict:
        return {
            "model": {"family": self.family, **self.model},
            "training": self.training,
            "loss": self.loss,
            "features": self.features,
        }

    def flat_params(self) -> dict[str, object]:
        """Flattened params for MLflow logging."""
        out: dict[str, object] = {"family": self.family}
        for section in ("model", "training", "loss", "features"):
            for k, v in getattr(self, section).items():
                out[f"{section}.{k}"] = v
        return out
