"""Deterministic analytics over the close dataset, backed by DuckDB.

Everything in this module is reproducible SQL. No model is involved. The control
layer calls these queries to establish *facts* (is there a twin posting? what did
this account do last month? is the reconciliation open?), and only those facts
plus retrieved policy text are ever put in front of a language model.

Keeping the analytics here rather than inside the workflow means the same numbers
can be recomputed and challenged by an auditor without running any inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from fcca.shared.config import Settings, get_settings
from fcca.shared.errors import DataNotFoundError

logger = logging.getLogger(__name__)


def build_close_database(settings: Settings | None = None) -> Path:
    """(Re)build ``close.duckdb`` from the generated CSV extracts."""
    settings = settings or get_settings()
    settings.ensure_directories()
    for path in (
        settings.journal_entries_path,
        settings.reconciliations_path,
        settings.exceptions_path,
    ):
        if not path.exists():
            raise DataNotFoundError(f"missing extract: {path}. Run `fcca generate-data` first.")

    db_path = settings.close_db_path
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        for table, source, types in (
            # Account numbers look numeric but are identifiers: force VARCHAR so
            # '199000' never silently becomes an integer and breaks a join or a
            # membership test against the high-risk account list.
            ("journal_entries", settings.journal_entries_path, "{'account': 'VARCHAR'}"),
            ("reconciliations", settings.reconciliations_path, "{'account': 'VARCHAR'}"),
            ("close_exceptions", settings.exceptions_path, None),
        ):
            options = "header=true, sample_size=-1" + (f", types={types}" if types else "")
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv(?, {options})",
                [str(source)],
            )
        con.execute("CREATE INDEX idx_je_journal ON journal_entries(journal_id)")
    finally:
        con.close()
    logger.info("built close database at %s", db_path)
    return db_path


@dataclass(frozen=True)
class VarianceResult:
    """Month-on-month movement for one account within one entity."""

    current_period_total: float
    prior_average: float
    absolute_change: float
    percent_change: float | None


class CloseAnalytics:
    """Read-only query surface over the close database.

    Use as a context manager, or call :meth:`close` explicitly.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.close_db_path.exists():
            raise DataNotFoundError(
                f"{self.settings.close_db_path} not found. Run `fcca generate-data` first."
            )
        self._con = duckdb.connect(str(self.settings.close_db_path), read_only=True)

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> CloseAnalytics:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._con.close()

    # -- helpers -----------------------------------------------------------
    def _one(self, sql: str, params: list[Any]) -> dict[str, Any] | None:
        cursor = self._con.execute(sql, params)
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [d[0] for d in cursor.description or []]
        return dict(zip(columns, row, strict=False))

    def _all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        cursor = self._con.execute(sql, params)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description or []]
        return [dict(zip(columns, row, strict=False)) for row in rows]

    # -- lookups -----------------------------------------------------------
    def journal_entry(self, journal_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM journal_entries WHERE journal_id = ?", [journal_id])
        if row is None:
            raise DataNotFoundError(f"journal entry {journal_id!r} not found")
        return row

    def exception(self, exception_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM close_exceptions WHERE exception_id = ?", [exception_id])
        if row is None:
            raise DataNotFoundError(f"exception {exception_id!r} not found")
        return row

    def exceptions(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM close_exceptions ORDER BY exception_id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self._all(sql, [])

    def reconciliation(self, company_code: str, account: str, period: str) -> dict[str, Any] | None:
        return self._one(
            "SELECT * FROM reconciliations WHERE company_code = ? AND account = ? AND period = ?",
            [company_code, account, period],
        )

    # -- analytical checks -------------------------------------------------
    def duplicate_candidates(self, journal_id: str) -> list[dict[str, Any]]:
        """Other entries matching company code, account, absolute amount and document date."""
        return self._all(
            """
            WITH target AS (SELECT * FROM journal_entries WHERE journal_id = ?)
            SELECT j.journal_id, j.user_id, j.posting_timestamp, j.amount, j.supporting_document
            FROM journal_entries j, target t
            WHERE j.journal_id <> t.journal_id
              AND j.company_code = t.company_code
              AND j.account = t.account
              AND j.document_date = t.document_date
              AND abs(j.amount - t.amount) < 0.005
            ORDER BY j.journal_id
            """,
            [journal_id],
        )

    def account_variance(self, company_code: str, account: str, period: str) -> VarianceResult:
        """Compare the period's account movement against the average of prior periods."""
        current = self._one(
            """
            SELECT COALESCE(sum(amount_reporting_ccy), 0) AS total
            FROM journal_entries
            WHERE company_code = ? AND account = ? AND strftime(posting_date, '%Y-%m') = ?
            """,
            [company_code, account, period],
        )
        prior = self._one(
            """
            SELECT COALESCE(avg(total), 0) AS prior_avg FROM (
                SELECT strftime(posting_date, '%Y-%m') AS p,
                       sum(amount_reporting_ccy) AS total
                FROM journal_entries
                WHERE company_code = ? AND account = ? AND strftime(posting_date, '%Y-%m') < ?
                GROUP BY p
            )
            """,
            [company_code, account, period],
        )
        current_total = float(current["total"]) if current else 0.0
        prior_avg = float(prior["prior_avg"]) if prior else 0.0
        change = current_total - prior_avg
        percent = (change / abs(prior_avg) * 100.0) if abs(prior_avg) > 1e-9 else None
        return VarianceResult(current_total, prior_avg, change, percent)

    def combination_history(
        self, company_code: str, account: str, cost_center: str, period: str
    ) -> int:
        """How often this account / cost centre pair was used in earlier periods."""
        row = self._one(
            """
            SELECT count(*) AS n FROM journal_entries
            WHERE company_code = ? AND account = ? AND cost_center = ?
              AND strftime(posting_date, '%Y-%m') < ?
            """,
            [company_code, account, cost_center, period],
        )
        return int(row["n"]) if row else 0

    def user_account_history(
        self, company_code: str, account: str, user_id: str, period: str
    ) -> int:
        """How often this user posted to this account in earlier periods."""
        row = self._one(
            """
            SELECT count(*) AS n FROM journal_entries
            WHERE company_code = ? AND account = ? AND user_id = ?
              AND strftime(posting_date, '%Y-%m') < ?
            """,
            [company_code, account, user_id, period],
        )
        return int(row["n"]) if row else 0

    def same_day_aggregate(self, journal_id: str) -> float:
        """Total reporting-currency amount posted to the same account, cost centre and day.

        Supports the anti-splitting rule in the Journal Entry Policy.
        """
        row = self._one(
            """
            WITH target AS (SELECT * FROM journal_entries WHERE journal_id = ?)
            SELECT COALESCE(sum(abs(j.amount_reporting_ccy)), 0) AS total
            FROM journal_entries j, target t
            WHERE j.company_code = t.company_code
              AND j.account = t.account
              AND j.cost_center = t.cost_center
              AND j.posting_date = t.posting_date
            """,
            [journal_id],
        )
        return float(row["total"]) if row else 0.0

    def dataset_summary(self) -> dict[str, Any]:
        row = self._one(
            """
            SELECT (SELECT count(*) FROM journal_entries)   AS journal_entries,
                   (SELECT count(*) FROM reconciliations)   AS reconciliations,
                   (SELECT count(*) FROM close_exceptions)  AS close_exceptions,
                   (SELECT count(DISTINCT company_code) FROM journal_entries) AS entities,
                   (SELECT count(DISTINCT account) FROM journal_entries)      AS accounts,
                   (SELECT count(DISTINCT user_id) FROM journal_entries)      AS users
            """,
            [],
        )
        return row or {}
