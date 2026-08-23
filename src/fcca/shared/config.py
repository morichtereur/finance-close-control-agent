"""Central configuration.

Design rule for this repository: **no model id, threshold, tolerance or approval
limit is hard-coded outside this module**. The values themselves live in
``config/thresholds.yaml``; this module gives them types, bounds and a single
place to read them from.

Why a YAML file rather than Python constants: a tolerance is a business rule.
The people who own it — a controller, a process owner, an internal auditor —
must be able to read the current values, propose a change, and see that change
reviewed, without any of that being a code change. A constant buried in a module
is a rule only a developer can find.

Why the values are still typed here: a YAML file alone will happily accept a
negative tolerance or a confidence threshold of 4.0. The bounds below are the
reason a bad edit fails at start-up instead of silently changing what the system
approves.

Precedence is environment variable > ``.env`` > ``config/thresholds.yaml`` >
field default, so an operator can override a single value for one run without
editing the file that everyone else reads.

Credentials are deliberately *not* settings. AWS and Google Cloud authentication
goes through the native credential chain of each SDK, so no secret ever has to
be typed into this project's configuration surface or captured in its audit log.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

ProviderName = Literal["mock", "bedrock", "vertex"]
StructuredOutputMode = Literal["json_schema_prompt", "native_tools"]

#: Repository root (``src/fcca/shared/config.py`` -> ``shared`` -> ``fcca`` ->
#: ``src`` -> root).
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]

#: The business-rule file. Overridable with ``FCCA_CONFIG_FILE`` so that tests
#: and alternative deployments can point at their own tolerances.
DEFAULT_CONFIG_FILE = REPO_ROOT / "config" / "thresholds.yaml"


class I2PConfig(BaseModel):
    """Invoice-to-pay tolerances, approval limits and routing rules.

    Nested rather than flat because these are one coherent set of rules that a
    process owner reads together, and because ``i2p.price_tolerance_pct`` says
    which process it governs in a way that ``price_tolerance_pct`` does not.
    """

    model_config = ConfigDict(frozen=True)

    # --------------------------------------------------------------- tolerances
    price_tolerance_pct: float = Field(
        default=2.0,
        ge=0.0,
        le=100.0,
        description="Permitted deviation of normalised net unit price, in percent.",
    )
    price_tolerance_abs: float = Field(
        default=25.0,
        ge=0.0,
        description=(
            "Permitted deviation per line in document currency. A line passes if it "
            "is inside EITHER limit — see tolerance_evaluation for why."
        ),
    )
    quantity_tolerance_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Permitted over-delivery relative to goods received, in percent. Zero by "
            "default: paying for more than was received is the failure this process exists "
            "to prevent."
        ),
    )
    gr_grace_days: int = Field(
        default=3,
        ge=0,
        description=(
            "Days an invoice may arrive ahead of its goods receipt before the missing "
            "receipt is treated as an exception rather than ordinary timing."
        ),
    )

    # ------------------------------------------------------------ duplicate check
    duplicate_window_days: int = Field(
        default=90,
        gt=0,
        description="Look-back window for duplicate candidates, in days.",
    )
    duplicate_amount_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        description="Amounts within this absolute difference count as equal.",
    )
    duplicate_reference_similarity: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity above which two vendor references are treated as the same "
            "reference. Fuzzy because 'INV-4471' and 'INV 4471' are the same document."
        ),
    )

    # ------------------------------------------------------------ routing limits
    auto_clear_max_value: float = Field(
        default=5_000.0,
        gt=0,
        description=(
            "Document value at or above which nothing is cleared without a person, "
            "however clean the match and however confident the model."
        ),
    )
    auto_clear_min_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Model confidence below which a proposal always goes to a person.",
    )
    propose_max_value: float = Field(
        default=50_000.0,
        gt=0,
        description=(
            "Document value at or above which an exception is escalated rather than "
            "offered to an approver as a proposal."
        ),
    )

    @model_validator(mode="after")
    def _limits_are_ordered(self) -> I2PConfig:
        """A proposal limit below the auto-clear limit would make the tiers unreachable."""
        if self.propose_max_value <= self.auto_clear_max_value:
            raise ValueError(
                "i2p.propose_max_value must exceed i2p.auto_clear_max_value; "
                f"got {self.propose_max_value} <= {self.auto_clear_max_value}"
            )
        return self


class Settings(BaseSettings):
    """Runtime configuration for both process modules.

    Values are read from environment variables (case-insensitive), a ``.env``
    file, and ``config/thresholds.yaml``, in that order of precedence. See
    ``.env.example`` for the environment surface and the YAML file itself for
    the business rules.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file=DEFAULT_CONFIG_FILE,
        yaml_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML source *below* the environment.

        Order is highest priority first. The YAML file holds the agreed business
        rules; an environment variable is a deliberate, run-scoped override of
        one of them, so it has to win. The audit trail records the values that
        were actually in force, not the file, which is what makes the override
        safe to allow at all.
        """
        import os

        yaml_path = Path(os.environ.get("FCCA_CONFIG_FILE", DEFAULT_CONFIG_FILE))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),
            file_secret_settings,
        )

    # ------------------------------------------------------------------ provider
    llm_provider: ProviderName = Field(
        default="mock",
        description="Active model provider. 'mock' requires no cloud account.",
    )

    # AWS Bedrock
    aws_region: str = Field(default="eu-central-1", description="Bedrock region.")
    bedrock_model_id: str = Field(
        default="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        description=(
            "Any chat model id enabled in the target Bedrock account. The "
            "workflow does not depend on a specific model family."
        ),
    )

    # Google Vertex AI
    google_cloud_project: str | None = Field(default=None, description="GCP project id.")
    vertex_location: str = Field(default="europe-west4", description="Vertex AI region.")
    vertex_model_name: str = Field(
        default="gemini-2.5-flash",
        description="Any chat model available to the project in Vertex AI.",
    )

    # Mock provider (labelled explicitly so it can never be mistaken for a real model)
    mock_model_name: str = Field(
        default="deterministic-stub-v1",
        validation_alias=AliasChoices("FCCA_MOCK_MODEL_NAME", "mock_model_name"),
        description="Identifier recorded in the audit trail for mock runs.",
    )

    # ----------------------------------------------------------------- inference
    temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices("FCCA_TEMPERATURE", "temperature"),
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=1200, validation_alias=AliasChoices("FCCA_MAX_TOKENS", "max_tokens"), gt=0
    )
    request_timeout_s: int = Field(
        default=60,
        validation_alias=AliasChoices("FCCA_REQUEST_TIMEOUT_S", "request_timeout_s"),
        gt=0,
    )
    structured_output_mode: StructuredOutputMode = Field(
        default="json_schema_prompt",
        validation_alias=AliasChoices("FCCA_STRUCTURED_OUTPUT_MODE", "structured_output_mode"),
        description=(
            "'json_schema_prompt' keeps the workflow portable across providers; "
            "'native_tools' delegates to the provider's own structured-output API."
        ),
    )
    max_parse_retries: int = Field(
        default=1,
        validation_alias=AliasChoices("FCCA_MAX_PARSE_RETRIES", "max_parse_retries"),
        ge=0,
        le=3,
    )

    # ----------------------------------------------------------- control thresholds
    materiality_group: float = Field(
        default=250_000.0,
        validation_alias=AliasChoices("FCCA_MATERIALITY_GROUP", "materiality_group"),
        gt=0,
        description="Group materiality for the close cycle, in reporting currency.",
    )
    journal_approval_threshold: float = Field(
        default=50_000.0,
        validation_alias=AliasChoices(
            "FCCA_JOURNAL_APPROVAL_THRESHOLD", "journal_approval_threshold"
        ),
        gt=0,
        description="Single-entry amount requiring documented second-level approval.",
    )
    trivial_threshold: float = Field(
        default=5_000.0,
        validation_alias=AliasChoices("FCCA_TRIVIAL_THRESHOLD", "trivial_threshold"),
        gt=0,
        description="Below this amount an item is not escalated on amount alone.",
    )
    business_hours_start: int = Field(
        default=7,
        validation_alias=AliasChoices("FCCA_BUSINESS_HOURS_START", "business_hours_start"),
        ge=0,
        le=23,
    )
    business_hours_end: int = Field(
        default=20,
        validation_alias=AliasChoices("FCCA_BUSINESS_HOURS_END", "business_hours_end"),
        ge=1,
        le=24,
    )
    late_posting_days: int = Field(
        default=5,
        validation_alias=AliasChoices("FCCA_LATE_POSTING_DAYS", "late_posting_days"),
        ge=0,
        description="Days between document date and posting date that count as late.",
    )
    high_risk_accounts: tuple[str, ...] = Field(
        default=("510000", "610000", "289000", "199000", "480000"),
        validation_alias=AliasChoices("FCCA_HIGH_RISK_ACCOUNTS", "high_risk_accounts"),
        description="Accounts flagged as inherently higher risk by the control catalogue.",
    )

    # ------------------------------------------------------------ human-in-the-loop
    auto_approve_min_confidence: float = Field(
        default=0.80,
        validation_alias=AliasChoices(
            "FCCA_AUTO_APPROVE_MIN_CONFIDENCE", "auto_approve_min_confidence"
        ),
        ge=0.0,
        le=1.0,
    )
    auto_approve_min_evidence: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "FCCA_AUTO_APPROVE_MIN_EVIDENCE", "auto_approve_min_evidence"
        ),
        ge=0,
    )

    # ------------------------------------------------------------ invoice-to-pay
    i2p: I2PConfig = Field(
        default_factory=lambda: I2PConfig(),
        description="Invoice-to-pay tolerances, approval limits and routing rules.",
    )

    # ------------------------------------------------------------------- retrieval
    retrieval_top_k: int = Field(
        default=4, validation_alias=AliasChoices("FCCA_RETRIEVAL_TOP_K", "retrieval_top_k"), gt=0
    )
    retrieval_min_score: float = Field(
        default=0.05,
        validation_alias=AliasChoices("FCCA_RETRIEVAL_MIN_SCORE", "retrieval_min_score"),
        ge=0.0,
    )
    chunk_size: int = Field(
        default=900, validation_alias=AliasChoices("FCCA_CHUNK_SIZE", "chunk_size"), gt=100
    )
    chunk_overlap: int = Field(
        default=120, validation_alias=AliasChoices("FCCA_CHUNK_OVERLAP", "chunk_overlap"), ge=0
    )

    # ---------------------------------------------------------------- data generation
    random_seed: int = Field(
        default=20_260_816, validation_alias=AliasChoices("FCCA_RANDOM_SEED", "random_seed")
    )
    n_journal_entries: int = Field(
        default=800,
        validation_alias=AliasChoices("FCCA_N_JOURNAL_ENTRIES", "n_journal_entries"),
        gt=0,
    )
    n_exceptions: int = Field(
        default=60, validation_alias=AliasChoices("FCCA_N_EXCEPTIONS", "n_exceptions"), gt=0
    )

    # ------------------------------------------------------------ cost estimation
    input_cost_per_mtok: float | None = Field(
        default=None,
        validation_alias=AliasChoices("FCCA_INPUT_COST_PER_MTOK", "input_cost_per_mtok"),
    )
    output_cost_per_mtok: float | None = Field(
        default=None,
        validation_alias=AliasChoices("FCCA_OUTPUT_COST_PER_MTOK", "output_cost_per_mtok"),
    )

    # ------------------------------------------------------------------------ paths
    base_dir: Path = Field(
        default=REPO_ROOT, validation_alias=AliasChoices("FCCA_BASE_DIR", "base_dir")
    )

    @field_validator("high_risk_accounts", mode="before")
    @classmethod
    def _split_accounts(cls, value: object) -> object:
        """Allow ``FCCA_HIGH_RISK_ACCOUNTS=510000,610000`` from the environment."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def policies_dir(self) -> Path:
        return self.base_dir / "policies"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_data_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evaluation_dir(self) -> Path:
        return self.base_dir / "data" / "evaluation"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def index_dir(self) -> Path:
        return self.processed_data_dir / "policy_index"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def audit_db_path(self) -> Path:
        return self.processed_data_dir / "audit.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def close_trace_path(self) -> Path:
        """Append-only step trace for the close module."""
        return self.processed_data_dir / "close_trace.jsonl"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def i2p_trace_path(self) -> Path:
        """Append-only step trace for the invoice-to-pay module."""
        return self.processed_data_dir / "i2p_trace.jsonl"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def i2p_data_dir(self) -> Path:
        """Structured JSON dataset for the invoice-to-pay module.

        JSON rather than CSV because invoices are nested documents — header,
        lines, cascading discounts — and flattening them into a table would be a
        modelling decision made for the storage format's convenience.
        """
        return self.raw_data_dir / "i2p"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def i2p_labels_path(self) -> Path:
        return self.evaluation_dir / "i2p_labels.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def i2p_results_path(self) -> Path:
        return self.processed_data_dir / "i2p_results.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def journal_entries_path(self) -> Path:
        return self.raw_data_dir / "journal_entries.csv"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exceptions_path(self) -> Path:
        return self.raw_data_dir / "close_exceptions.csv"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def reconciliations_path(self) -> Path:
        return self.raw_data_dir / "reconciliations.csv"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def close_db_path(self) -> Path:
        return self.processed_data_dir / "close.duckdb"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def labelled_set_path(self) -> Path:
        return self.evaluation_dir / "labelled_exceptions.json"

    def model_name_for(self, provider: ProviderName | None = None) -> str:
        """Return the configured model identifier for ``provider``.

        This is the single place that maps a provider to a model id; nothing
        downstream needs to know which cloud is active.
        """
        provider = provider or self.llm_provider
        match provider:
            case "bedrock":
                return self.bedrock_model_id
            case "vertex":
                return self.vertex_model_name
            case "mock":
                return self.mock_model_name
        raise ValueError(f"Unknown provider: {provider!r}")

    def ensure_directories(self) -> None:
        """Create the writable directories used by the CLI."""
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.evaluation_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_snapshot(self) -> dict[str, object]:
        """Non-secret settings snapshot recorded with every audit entry.

        Only decision-relevant configuration is included, so an auditor can see
        which thresholds were in force when a recommendation was produced.
        """
        return {
            "materiality_group": self.materiality_group,
            "journal_approval_threshold": self.journal_approval_threshold,
            "trivial_threshold": self.trivial_threshold,
            "late_posting_days": self.late_posting_days,
            "business_hours": [self.business_hours_start, self.business_hours_end],
            "high_risk_accounts": list(self.high_risk_accounts),
            "auto_approve_min_confidence": self.auto_approve_min_confidence,
            "auto_approve_min_evidence": self.auto_approve_min_evidence,
            "retrieval_top_k": self.retrieval_top_k,
            "structured_output_mode": self.structured_output_mode,
            "temperature": self.temperature,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests that patch the environment)."""
    get_settings.cache_clear()
