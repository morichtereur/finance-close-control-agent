"""Tests for the shared append-only trace and the YAML configuration layer.

The trace is the product, so its guarantees are tested as product behaviour, not
as implementation detail: a record must name its actor honestly, a rule record
must not carry model provenance, the file must only ever grow, and the same
inputs must hash the same way.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from fcca.shared.config import Settings
from fcca.shared.trace import (
    TraceRecord,
    TraceWriter,
    hash_input,
    read_trace,
    render_trace,
)


def _record(**overrides: object) -> TraceRecord:
    defaults: dict[str, object] = {
        "timestamp": datetime(2026, 8, 1, 9, 30, tzinfo=UTC),
        "case_id": "INV-0001",
        "module": "i2p",
        "step_name": "price_check",
        "actor": "rule",
        "input_hash": "abc123",
        "outcome": "pass",
        "summary": "Normalised unit price matches the purchase order.",
        "rule_id": "I2P-R-010",
    }
    defaults.update(overrides)
    return TraceRecord.model_validate(defaults)


# --------------------------------------------------------------------- schema
class TestProvenance:
    """A record may not claim one actor while carrying another's provenance."""

    def test_rule_record_requires_a_rule_id(self) -> None:
        with pytest.raises(ValidationError, match="requires a rule_id"):
            _record(rule_id=None)

    def test_rule_record_rejects_model_provenance(self) -> None:
        with pytest.raises(ValidationError, match="must not carry model provenance"):
            _record(model="claude-sonnet-4-5", prompt_version="i2p-v1")

    def test_model_record_requires_model_and_prompt_version(self) -> None:
        with pytest.raises(ValidationError, match="requires both model and prompt_version"):
            _record(actor="model", rule_id=None, model="claude-sonnet-4-5")

    def test_model_record_rejects_a_rule_id(self) -> None:
        with pytest.raises(ValidationError, match="must not carry a rule_id"):
            _record(
                actor="model",
                rule_id="I2P-R-010",
                model="claude-sonnet-4-5",
                prompt_version="i2p-v1",
            )

    def test_human_record_carries_no_machine_provenance(self) -> None:
        with pytest.raises(ValidationError, match="must not carry rule or model provenance"):
            _record(actor="human")

    def test_valid_records_of_each_actor(self) -> None:
        assert _record().provenance == "I2P-R-010"
        model_record = _record(
            actor="model", rule_id=None, model="claude-sonnet-4-5", prompt_version="i2p-v1"
        )
        assert model_record.provenance == "claude-sonnet-4-5@i2p-v1"
        assert _record(actor="human", rule_id=None).provenance == "human"


# ---------------------------------------------------------------- input hashes
class TestInputHash:
    def test_same_input_hashes_the_same_regardless_of_key_order(self) -> None:
        assert hash_input({"a": 1, "b": 2}) == hash_input({"b": 2, "a": 1})

    def test_different_input_hashes_differently(self) -> None:
        assert hash_input({"amount": 100.0}) != hash_input({"amount": 100.01})

    def test_hash_is_short_enough_to_compare_by_eye(self) -> None:
        assert len(hash_input({"x": 1})) == 16


# --------------------------------------------------------------- append-only
class TestTraceWriter:
    def test_appends_one_line_per_record(self, tmp_path: Path) -> None:
        writer = TraceWriter(tmp_path / "trace.jsonl", module="i2p")
        for step in ("intake", "price_check", "routing_decision"):
            writer.step(
                case_id="INV-0001",
                step_name=step,
                actor="rule",
                rule_id=f"I2P-{step}",
                inputs={"step": step},
                outcome="pass",
                summary=f"{step} completed.",
            )
        lines = (tmp_path / "trace.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["step_name"] for line in lines] == [
            "intake",
            "price_check",
            "routing_decision",
        ]

    def test_a_second_writer_appends_rather_than_truncating(self, tmp_path: Path) -> None:
        """The file only ever grows. This is the property the whole design rests on."""
        path = tmp_path / "trace.jsonl"
        TraceWriter(path, module="i2p").append(_record())
        TraceWriter(path, module="i2p").append(_record(step_name="tolerance_evaluation"))
        assert len(read_trace(path)) == 2

    def test_writer_exposes_no_mutation_api(self) -> None:
        """There is deliberately no update or delete. A trace you can edit proves nothing."""
        for forbidden in ("update", "delete", "remove", "truncate", "rewrite"):
            assert not hasattr(TraceWriter, forbidden)

    def test_creates_the_parent_directory(self, tmp_path: Path) -> None:
        writer = TraceWriter(tmp_path / "nested" / "deeper" / "trace.jsonl", module="close")
        writer.append(_record(module="close"))
        assert writer.path.exists()


# -------------------------------------------------------------------- reading
class TestReadTrace:
    def test_missing_file_reads_as_empty_rather_than_raising(self, tmp_path: Path) -> None:
        assert read_trace(tmp_path / "absent.jsonl") == []

    def test_filters_to_one_case(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        writer = TraceWriter(path, module="i2p")
        writer.append(_record(case_id="INV-0001"))
        writer.append(_record(case_id="INV-0002"))
        writer.append(_record(case_id="INV-0001", step_name="routing_decision"))
        assert [r.step_name for r in read_trace(path, case_id="INV-0001")] == [
            "price_check",
            "routing_decision",
        ]

    def test_round_trips_through_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        original = _record(detail={"residual_pct": 0.0, "tolerance_pct": 2.0})
        TraceWriter(path, module="i2p").append(original)
        (restored,) = read_trace(path)
        assert restored.detail == original.detail
        assert restored.timestamp == original.timestamp

    def test_render_is_readable_and_names_the_actor(self) -> None:
        text = render_trace([_record(), _record(step_name="routing_decision")])
        assert "price_check" in text
        assert "rule" in text
        assert len(text.splitlines()) == 2

    def test_render_handles_an_empty_trace(self) -> None:
        assert render_trace([]) == "(no trace records)"


# ------------------------------------------------------------- YAML config
class TestYamlConfiguration:
    """Tolerances are business rules, so they must be readable and overridable."""

    @staticmethod
    def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
        """Point FCCA_CONFIG_FILE at a throwaway rule file."""
        config = tmp_path / "thresholds.yaml"
        config.write_text(body)
        monkeypatch.setenv("FCCA_CONFIG_FILE", str(config))

    def test_values_come_from_the_yaml_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._config(tmp_path, monkeypatch, "materiality_group: 999000.0\nlate_posting_days: 11\n")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.materiality_group == 999_000.0
        assert settings.late_posting_days == 11

    def test_environment_overrides_the_yaml_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator can override one rule for one run without editing the shared file."""
        self._config(tmp_path, monkeypatch, "materiality_group: 999000.0\n")
        monkeypatch.setenv("FCCA_MATERIALITY_GROUP", "123000")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.materiality_group == 123_000.0

    def test_an_out_of_bounds_value_fails_at_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A YAML file will accept anything; the point of the typed layer is that we do not."""
        self._config(tmp_path, monkeypatch, "auto_approve_min_confidence: 4.0\n")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_the_shipped_config_file_parses_and_is_in_force(self) -> None:
        """The committed file must actually load — a broken one would fail silently to defaults."""
        import yaml

        from fcca.shared.config import DEFAULT_CONFIG_FILE

        assert DEFAULT_CONFIG_FILE.exists()
        data = yaml.safe_load(DEFAULT_CONFIG_FILE.read_text())
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.materiality_group == data["materiality_group"]
        assert list(settings.high_risk_accounts) == data["high_risk_accounts"]
