"""Declarative feature / fit specification for the tabular booster families.

``popularity`` and ``sasrec`` have a fixed input shape, so a configuration is
enough to describe an experiment.  The gradient-boosting families do not: what
goes into the model *is* the modeling decision.  That decision belongs to the
Planner and must be approved by a human before a job is set up, so it is
expressed here as a closed schema rather than as agent-authored Python: every
transform, label rule and candidate rule is enumerable, hashable and replayable.

``feature_fit_hash`` is the identity a human approves. It deliberately excludes
``FitSpec.booster_params`` so the Trainer can keep tuning hyperparameters inside
an approved shape without asking again; anything structural changes the hash and
returns the campaign to the checkpoint.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from amt.schemas.common import AmtModel, new_id, stable_hash

TABULAR_FAMILIES: frozenset[str] = frozenset(
    {
        "xgboost_classifier",
        "xgboost_reranker",
        "lightgbm_classifier",
        "lightgbm_reranker",
    }
)
RANKER_FAMILIES: frozenset[str] = frozenset(
    {"xgboost_reranker", "lightgbm_reranker"}
)

TransformKind = Literal["column", "cast", "bucket", "hash", "count", "ratio", "literal"]
CountScope = Literal["train", "history"]
CastTarget = Literal["float", "int", "code"]
FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "not_null"]
SourceRole = Literal["interactions", "item_attributes"]
LabelStrategy = Literal["holdout_next_item", "interaction_positive", "label_column"]
CandidateStrategy = Literal["popularity_topn", "history_attribute_expansion"]

# Columns the runtime always puts on a candidate row, so a transform may read
# them without declaring a source.
BASE_COLUMNS: tuple[str, ...] = (
    "item_id",
    "history_length",
    "item_train_count",
    "item_train_rank",
)


class RecipeSource(AmtModel):
    """One input to the recipe: a frozen split, or a table named by the Planner."""

    name: str
    role: SourceRole = "interactions"
    split: Literal["train", "validation"] | None = Field(
        default=None, description="Frozen split to read through SplitGuard. Never 'qa'."
    )
    table: str | None = Field(
        default=None, description="Fully-qualified UC table, for feature attributes only."
    )
    columns: list[str] = Field(default_factory=list)
    join_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one_origin(self) -> RecipeSource:
        if bool(self.split) == bool(self.table):
            raise ValueError(
                f"source {self.name!r} must name exactly one of split / table"
            )
        if self.table and self.role != "item_attributes":
            raise ValueError(
                f"source {self.name!r}: external tables may only supply item_attributes; "
                "the labelled population comes from the frozen splits"
            )
        return self


class RecipeJoin(AmtModel):
    """How a source is attached to the candidate rows."""

    source: str = Field(description="``RecipeSource.name`` being joined in.")
    keys: list[str] = Field(min_length=1)
    how: Literal["left", "inner"] = "left"


class RecipeFilter(AmtModel):
    """A closed-form row filter. No free-form SQL is executed."""

    column: str
    op: FilterOp
    value: Any = None

    @model_validator(mode="after")
    def _value_required(self) -> RecipeFilter:
        if self.op == "not_null":
            return self
        if self.value is None:
            raise ValueError(f"filter on {self.column!r} with op {self.op!r} needs a value")
        if self.op in {"in", "not_in"} and not isinstance(self.value, (list, tuple)):
            raise ValueError(f"filter on {self.column!r} with op {self.op!r} needs a list value")
        return self


class FeatureTransform(AmtModel):
    """One numeric feature column, built by one of the supported transforms.

    * ``column``   -- take an existing column (strings are category-encoded at fit)
    * ``cast``     -- coerce to float / int / a stable category code
    * ``bucket``   -- bucketize a numeric column on explicit boundaries
    * ``hash``     -- stable hash of a column into ``buckets`` values
    * ``count``    -- frequency of ``of`` over the train split or the user's history
    * ``ratio``    -- safe division of two already-defined features
    * ``literal``  -- a constant
    """

    name: str
    kind: TransformKind
    source: str | None = Field(default=None, description="Input column for column/cast/bucket/hash.")
    of: str | None = Field(default=None, description="Counted column for kind='count'.")
    over: CountScope | None = Field(default=None, description="Count scope for kind='count'.")
    numerator: str | None = None
    denominator: str | None = None
    to: CastTarget | None = None
    boundaries: list[float] = Field(default_factory=list)
    buckets: int | None = Field(default=None, ge=2)
    value: float | None = None

    @model_validator(mode="after")
    def _kind_requirements(self) -> FeatureTransform:
        required = {
            "column": ("source",),
            "cast": ("source", "to"),
            "bucket": ("source", "boundaries"),
            "hash": ("source", "buckets"),
            "count": ("of", "over"),
            "ratio": ("numerator", "denominator"),
            "literal": ("value",),
        }[self.kind]
        for field in required:
            if not getattr(self, field):
                raise ValueError(f"transform {self.name!r} of kind {self.kind!r} needs {field!r}")
        return self


class LabelSpec(AmtModel):
    """How a positive row is defined. Never changes the contract's labels."""

    strategy: LabelStrategy = "holdout_next_item"
    source_column: str | None = Field(
        default=None, description="Binary label column, for strategy='label_column'."
    )
    positive_values: list[Any] = Field(
        default_factory=list, description="Values of ``source_column`` counted as positive."
    )
    definition: str = ""

    @model_validator(mode="after")
    def _column_strategy_needs_a_column(self) -> LabelSpec:
        if self.strategy == "label_column" and not self.source_column:
            raise ValueError("label strategy 'label_column' requires source_column")
        return self


class CandidateGeneration(AmtModel):
    """How the scored candidate set is built, at fit time and at serving time."""

    strategy: CandidateStrategy = "popularity_topn"
    n: int = Field(default=200, ge=1)
    exclude_seen: bool = True
    attribute: str | None = Field(
        default=None, description="Attribute expanded for 'history_attribute_expansion'."
    )

    @model_validator(mode="after")
    def _expansion_needs_an_attribute(self) -> CandidateGeneration:
        if self.strategy == "history_attribute_expansion" and not self.attribute:
            raise ValueError(
                "candidate strategy 'history_attribute_expansion' requires an attribute"
            )
        return self


class FeatureRecipe(AmtModel):
    """Everything about the model's input shape that a human has to agree to."""

    recipe_id: str = Field(default_factory=lambda: new_id("recipe"))
    row_grain: Literal["user_item_candidate"] = "user_item_candidate"
    sources: list[RecipeSource] = Field(default_factory=list)
    joins: list[RecipeJoin] = Field(default_factory=list)
    feature_columns: list[FeatureTransform] = Field(default_factory=list, min_length=1)
    label: LabelSpec = Field(default_factory=LabelSpec)
    group_key: str = Field(default="user_id", description="Query grouping for the rankers.")
    candidate_generation: CandidateGeneration = Field(default_factory=CandidateGeneration)
    filters: list[RecipeFilter] = Field(default_factory=list)
    leakage_notes: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _joins_reference_declared_sources(self) -> FeatureRecipe:
        names = {s.name for s in self.sources}
        unknown = [j.source for j in self.joins if j.source not in names]
        if unknown:
            raise ValueError(f"joins reference undeclared sources: {sorted(unknown)}")
        distinct = {t.name for t in self.feature_columns}
        if len(distinct) != len(self.feature_columns):
            raise ValueError("feature_columns contain duplicate names")
        return self

    @property
    def structural_payload(self) -> dict:
        """The recipe minus identifiers and prose — what approval actually covers."""
        payload = self.model_dump(
            mode="json", exclude={"recipe_id", "notes", "leakage_notes"}
        )
        label = payload.get("label")
        if isinstance(label, dict):
            label.pop("definition", None)
        return payload

    @property
    def recipe_hash(self) -> str:
        return stable_hash(self.structural_payload)


class FitSpec(AmtModel):
    """Booster fitting contract. ``booster_params`` is the Trainer's playground."""

    objective: str = Field(default="", description="Empty means the family default.")
    num_boost_round: int = Field(default=200, ge=1)
    early_stopping_rounds: int = Field(default=20, ge=0)
    eval_metric: str = ""
    negative_samples_per_positive: int = Field(default=4, ge=0)
    seed: int = 42
    booster_params: dict = Field(
        default_factory=dict,
        description="Hyperparameters the Trainer may change without re-approval.",
    )

    @property
    def structural_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"booster_params"})


def feature_fit_hash(recipe: FeatureRecipe | None, fit_spec: FitSpec | None) -> str:
    """Identity of an approved feature/fit shape (hyperparameters excluded)."""
    if recipe is None and fit_spec is None:
        return ""
    return stable_hash(
        {
            "recipe": recipe.structural_payload if recipe else None,
            "fit": fit_spec.structural_payload if fit_spec else None,
        }
    )


class FeatureFitProposal(AmtModel):
    """Planner output presented at the feature/fit human checkpoint."""

    family: str = Field(description="Tabular model family this shape is for.")
    feature_recipe: FeatureRecipe
    fit_spec: FitSpec = Field(default_factory=FitSpec)
    rationale: str = ""
    data_notes: str = ""
    leakage_argument: str = Field(
        default="", description="Why no feature can see the label's future."
    )

    @model_validator(mode="after")
    def _family_must_be_tabular(self) -> FeatureFitProposal:
        if self.family not in TABULAR_FAMILIES:
            raise ValueError(
                f"family {self.family!r} is not a tabular family; "
                f"expected one of {sorted(TABULAR_FAMILIES)}"
            )
        return self

    @property
    def feature_fit_hash(self) -> str:
        return feature_fit_hash(self.feature_recipe, self.fit_spec)


class ApprovedShape(AmtModel):
    """One hypothesis's feature/fit shape, as a human approved it at HC_FEATURE.

    The campaign keeps one of these per tabular hypothesis, approved before any
    hypothesis is planned. ``planner_select_hypothesis`` copies the selected
    one into the state keys the Trainer and the executor already read, so
    nothing downstream of selection knows the gate became batched.
    """

    hypothesis_id: str
    family: str
    feature_recipe: FeatureRecipe
    fit_spec: FitSpec = Field(default_factory=FitSpec)
    approved_at_hash: str = Field(
        default="", description="contract_hash the shape was approved under."
    )

    @property
    def feature_fit_hash(self) -> str:
        return feature_fit_hash(self.feature_recipe, self.fit_spec)


def is_tabular_family(family: str | None) -> bool:
    return str(family or "").strip().lower() in TABULAR_FAMILIES


def is_ranker_family(family: str | None) -> bool:
    return str(family or "").strip().lower() in RANKER_FAMILIES
