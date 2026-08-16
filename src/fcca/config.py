"""Central configuration.

Design rule for this repository: **no model id, threshold or path is hard-coded
outside this module**. Everything that an operator would plausibly want to change
between environments (provider, model, materiality, gate thresholds) is a
setting, readable from the environment or a local ``.env`` file.

Credentials are deliberately *not* settings. AWS and Google Cloud authentication
goes through the native credential chain of each SDK, so no secret ever has to
be typed into this project's configuration surface or captured in its audit log.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["mock", "bedrock", "vertex"]
StructuredOutputMode = Literal["json_schema_prompt", "native_tools"]

#: Repository root (``src/fcca/config.py`` -> ``src/fcca`` -> ``src`` -> root).
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


class Settings(BaseSettings):
    """Runtime configuration for the control agent.

    Attributes are populated from environment variables (case-insensitive) and
    from a ``.env`` file if present. See ``.env.example`` for the full surface.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
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
        alias="FCCA_MOCK_MODEL_NAME",
        description="Identifier recorded in the audit trail for mock runs.",
    )

    # ----------------------------------------------------------------- inference
    temperature: float = Field(default=0.0, alias="FCCA_TEMPERATURE", ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, alias="FCCA_MAX_TOKENS", gt=0)
    request_timeout_s: int = Field(default=60, alias="FCCA_REQUEST_TIMEOUT_S", gt=0)
    structured_output_mode: StructuredOutputMode = Field(
        default="json_schema_prompt",
        alias="FCCA_STRUCTURED_OUTPUT_MODE",
        description=(
            "'json_schema_prompt' keeps the workflow portable across providers; "
            "'native_tools' delegates to the provider's own structured-output API."
        ),
    )
    max_parse_retries: int = Field(default=1, alias="FCCA_MAX_PARSE_RETRIES", ge=0, le=3)

    # ----------------------------------------------------------- control thresholds
    materiality_group: float = Field(
        default=250_000.0,
        alias="FCCA_MATERIALITY_GROUP",
        gt=0,
        description="Group materiality for the close cycle, in reporting currency.",
    )
    journal_approval_threshold: float = Field(
        default=50_000.0,
        alias="FCCA_JOURNAL_APPROVAL_THRESHOLD",
        gt=0,
        description="Single-entry amount requiring documented second-level approval.",
    )
    trivial_threshold: float = Field(
        default=5_000.0,
        alias="FCCA_TRIVIAL_THRESHOLD",
        gt=0,
        description="Below this amount an item is not escalated on amount alone.",
    )
    business_hours_start: int = Field(default=7, alias="FCCA_BUSINESS_HOURS_START", ge=0, le=23)
    business_hours_end: int = Field(default=20, alias="FCCA_BUSINESS_HOURS_END", ge=1, le=24)
    late_posting_days: int = Field(
        default=5,
        alias="FCCA_LATE_POSTING_DAYS",
        ge=0,
        description="Days between document date and posting date that count as late.",
    )
    high_risk_accounts: tuple[str, ...] = Field(
        default=("510000", "610000", "289000", "199000", "480000"),
        alias="FCCA_HIGH_RISK_ACCOUNTS",
        description="Accounts flagged as inherently higher risk by the control catalogue.",
    )

    # ------------------------------------------------------------ human-in-the-loop
    auto_approve_min_confidence: float = Field(
        default=0.80, alias="FCCA_AUTO_APPROVE_MIN_CONFIDENCE", ge=0.0, le=1.0
    )
    auto_approve_min_evidence: int = Field(default=1, alias="FCCA_AUTO_APPROVE_MIN_EVIDENCE", ge=0)

    # ------------------------------------------------------------------- retrieval
    retrieval_top_k: int = Field(default=4, alias="FCCA_RETRIEVAL_TOP_K", gt=0)
    retrieval_min_score: float = Field(default=0.05, alias="FCCA_RETRIEVAL_MIN_SCORE", ge=0.0)
    chunk_size: int = Field(default=900, alias="FCCA_CHUNK_SIZE", gt=100)
    chunk_overlap: int = Field(default=120, alias="FCCA_CHUNK_OVERLAP", ge=0)

    # ---------------------------------------------------------------- data generation
    random_seed: int = Field(default=20_260_816, alias="FCCA_RANDOM_SEED")
    n_journal_entries: int = Field(default=800, alias="FCCA_N_JOURNAL_ENTRIES", gt=0)
    n_exceptions: int = Field(default=60, alias="FCCA_N_EXCEPTIONS", gt=0)

    # ------------------------------------------------------------ cost estimation
    input_cost_per_mtok: float | None = Field(default=None, alias="FCCA_INPUT_COST_PER_MTOK")
    output_cost_per_mtok: float | None = Field(default=None, alias="FCCA_OUTPUT_COST_PER_MTOK")

    # ------------------------------------------------------------------------ paths
    base_dir: Path = Field(default=REPO_ROOT, alias="FCCA_BASE_DIR")

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
