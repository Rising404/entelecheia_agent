"""Maintenance lifecycle reports remain visible without leaking document content."""

from collections.abc import Callable
import logging

import pytest

from personagraph.workspace.ingestion.lifecycle import DocumentMaintenanceWorkerLifecycle
from personagraph.workspace.ingestion.storage import DocumentMaintenanceRunReport


_LOGGER_NAME = "personagraph.workspace.ingestion.lifecycle"
_PRIVATE_CONTEXT = "private-document-body /private/customer/report.pdf fake-secret-token"


class _ReportingWorker:
    def __init__(
        self,
        outcomes: list[DocumentMaintenanceRunReport | Exception],
    ) -> None:
        self.outcomes = outcomes
        self.operations: list[str] = []
        self.run_count = 0
        self.drain_count = 0
        self.stop: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return _PRIVATE_CONTEXT

    def drain_outbox_once(self, *, limit: int | None = None) -> int:
        self.operations.append("drain")
        self.drain_count += 1
        if self.drain_count == 2 * len(self.outcomes):
            self.stop()
        return 0

    def run_once(self, *, limit: int = 16) -> DocumentMaintenanceRunReport:
        self.operations.append(f"run:{limit}")
        outcome = self.outcomes[self.run_count]
        self.run_count += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _run_passes(
    outcomes: list[DocumentMaintenanceRunReport | Exception],
) -> tuple[DocumentMaintenanceWorkerLifecycle, _ReportingWorker]:
    worker = _ReportingWorker(outcomes)
    lifecycle = DocumentMaintenanceWorkerLifecycle(worker, run_limit=7)
    worker.stop = lifecycle._stop_event.set
    # Exercise the actual pass loop synchronously. The queued wake makes a second
    # pass immediate; the last post-run drain stops it without wall-clock sleeps.
    lifecycle.wake()
    lifecycle._run_scoped()
    return lifecycle, worker


@pytest.mark.parametrize(
    "report",
    [
        DocumentMaintenanceRunReport(claimed=3, applied=2, retryable_failed=1),
        DocumentMaintenanceRunReport(claimed=3, applied=2, terminal_failed=1),
        DocumentMaintenanceRunReport(claimed=3, applied=2, lease_lost=1),
        DocumentMaintenanceRunReport(
            claimed=8,
            applied=2,
            retryable_failed=1,
            terminal_failed=2,
            lease_lost=3,
        ),
    ],
    ids=["retryable-failure", "terminal-failure", "lost-lease", "mixed-outcome"],
)
def test_incomplete_run_report_warns_without_stopping_drain_or_next_pass(
    report,
    caplog,
):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        lifecycle, worker = _run_passes(
            [report, DocumentMaintenanceRunReport(claimed=1, applied=1)]
        )

    assert worker.operations == ["drain", "run:7", "drain"] * 2
    assert lifecycle.passes == 2
    assert lifecycle.failures == 0
    assert lifecycle.last_error_type is None
    records = [record for record in caplog.records if record.name == _LOGGER_NAME]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "document_maintenance_run_incomplete "
        f"claimed={report.claimed} applied={report.applied} "
        f"retryable_failed={report.retryable_failed} "
        f"terminal_failed={report.terminal_failed} lease_lost={report.lease_lost}"
    )
    assert {
        name: value
        for name, value in vars(record).items()
        if name.startswith("maintenance_")
    } == {
        "maintenance_claimed": report.claimed,
        "maintenance_applied": report.applied,
        "maintenance_retryable_failed": report.retryable_failed,
        "maintenance_terminal_failed": report.terminal_failed,
        "maintenance_lease_lost": report.lease_lost,
    }
    assert record.args == (
        report.claimed,
        report.applied,
        report.retryable_failed,
        report.terminal_failed,
        report.lease_lost,
    )
    assert all(type(value) is int for value in record.args)
    assert record.exc_info is None
    assert _PRIVATE_CONTEXT not in repr(vars(record))


@pytest.mark.parametrize(
    "report",
    [
        DocumentMaintenanceRunReport(),
        DocumentMaintenanceRunReport(claimed=3, applied=3),
    ],
    ids=["idle", "all-applied"],
)
def test_healthy_run_report_does_not_warn(report, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        lifecycle, worker = _run_passes([report])

    assert worker.operations == ["drain", "run:7", "drain"]
    assert lifecycle.passes == 1
    assert lifecycle.failures == 0
    assert not [record for record in caplog.records if record.name == _LOGGER_NAME]


def test_escaped_exception_keeps_type_only_failure_observation(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        lifecycle, worker = _run_passes([RuntimeError(_PRIVATE_CONTEXT)])

    assert worker.operations == ["drain", "run:7", "drain"]
    assert lifecycle.passes == 0
    assert lifecycle.failures == 1
    assert lifecycle.last_error_type == "RuntimeError"
    assert _PRIVATE_CONTEXT not in caplog.text


def test_escaped_exception_does_not_stop_next_pass(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        lifecycle, worker = _run_passes(
            [RuntimeError(_PRIVATE_CONTEXT), DocumentMaintenanceRunReport()]
        )

    assert worker.operations == ["drain", "run:7", "drain"] * 2
    assert lifecycle.passes == 1
    assert lifecycle.failures == 1
    assert lifecycle.last_error_type is None
    assert _PRIVATE_CONTEXT not in caplog.text


def test_publication_ack_runs_after_outbox_without_a_new_worker():
    worker = _ReportingWorker([DocumentMaintenanceRunReport()])
    worker.stop = lambda: None

    def acknowledge():
        worker.operations.append("ack")
        lifecycle._stop_event.set()

    lifecycle = DocumentMaintenanceWorkerLifecycle(worker, after_pass=acknowledge)
    lifecycle._run_scoped()
    assert worker.operations == ["drain", "run:16", "drain", "ack"]
    assert lifecycle.passes == 1
