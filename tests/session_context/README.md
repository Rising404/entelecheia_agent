# SessionContext tests

This tree owns all tests whose primary subject is SessionContext. Tests are
grouped by capability so a maintainer can review or replace one concern without
searching through generic session, context, and runtime suites.

## Layout

- `contracts/`: typed contracts, catalog, reducer policy.
- `extraction/`: model JSON parsing and evidence ownership.
- `lifecycle/`: TTL, expiry, retract, and conflict lifecycle.
- `persistence/`: current evidence, candidate/view/audit, transactions, and durable reset.
- `runtime/`: SessionContext reset, inspection, and Repair API contracts.
- `maintenance/`: Repair, correction, diff, atomic replacement, reset-boundary replay.

## Commands

```bash
.venv/bin/pytest tests/session_context -q
.venv/bin/pytest tests/session_context/contracts tests/session_context/persistence -q
.venv/bin/pytest tests/session_context/lifecycle -q
```

Do not make a failing test pass by weakening its expected behavior without a
separate specification decision.
