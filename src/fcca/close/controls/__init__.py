"""Deterministic finance control checks.

Every check is a pure function of facts and configuration. None of them calls a
model. They produce :class:`~fcca.close.models.ControlSignal` objects that are stored
verbatim in the audit trail, so a reviewer can see exactly which rule fired and
against which threshold — independently of anything the model later said.
"""

from fcca.close.controls.engine import (
    mandatory_escalation_triggers,
    run_controls,
    to_journal_entry,
)
from fcca.close.controls.materiality import (
    amount_in_reporting_currency,
    check_materiality,
    materiality_band,
)

__all__ = [
    "amount_in_reporting_currency",
    "check_materiality",
    "mandatory_escalation_triggers",
    "materiality_band",
    "run_controls",
    "to_journal_entry",
]
