# DocBench evaluation state isolation

Entelecheia backend evaluations never use the GUI/API Session databases or its Project namespace. The supported unit is one case attempt, one fresh subprocess, one private state root and one Host-managed Session Project.

## External root

From the repository root, set an explicit absolute `PERSONAGRAPH_BENCH_EVAL_DIR` outside the checkout **before starting Python**, then invoke `python -m evals.docbench.reproduce_or_run_script`. No external launcher is needed. The environment variable names the parent of:

```text
docbench/
  source/                read-only benchmark inputs; never Agent-visible
  runs/<run-id>/         private case inputs, state, trajectory and scoring
  workspaces/<run-id>/   allocation root for Agent-visible Session Projects
```

## Required process boundary

For an initial attempt, the parent assigns `runs/<run-id>/cases/<case-id>/state/`. Retries receive fresh state below that case's `attempts/` directory. Before the child interpreter starts, the parent sets the exact case state, case-local config, benchmark root and run workspace in its environment.

Runtime path constants are frozen at import time. The worker checks the imported canonical paths before any state write or Provider work; a late environment assignment is not isolation. Private case state includes its project catalog/Session database, parsed chunks/indexes, turns/tasks, runtime trajectory, tool/model ledgers and authority/quota records.

## Project allocation

Workers in a run receive `workspaces/<run-id>` as `PERSONAGRAPH_DEFAULT_PROJECTS_DIR`. The product Session service, not a parallel evaluation implementation, creates and binds the actual Project below it. Every case creates a separate Session; retries create fresh Projects.

The runner checks that the Project is below its run workspace and records a DocBench-root-relative locator in private case output. `source/` and private `runs/` must never be bound to a Session. This prevents tools from reading other benchmark inputs, QA/reference answers, judge evidence or SQLite authority merely by exploring the Project directory.

## Configuration reuse is not state sharing

The parent may read the active installation Provider profiles to resolve and freeze endpoint identity. Case workers replace product state/config roots before interpreter startup and never inherit the GUI default Project root. For evaluation-only credentials, set `PERSONAGRAPH_LOCAL_CONFIG_DIR` to a separate absolute private directory before starting the parent.

Do not move real profiles, `.env` files, state databases or GUI Projects into the repository to make setup convenient. Session names are not an isolation boundary: database roots and Project authority are. Provider requests may still transmit authorized document content externally; process isolation is not a no-egress guarantee.

## Regression obligations

1. Child state is an exact descendant of the run's case tree.
2. Imported state/config constants equal the parent-owned case paths.
3. Imported default Project root equals the run workspace.
4. Workspace and private run/state paths do not overlap.
5. Created Session Project is below the run workspace.
6. Every retry uses fresh state and a fresh Host-created Project.
7. GUI catalog/Session databases receive no benchmark records.

Preserve these checks in `tests/evals/`. See the [runbook](formal_l1_eval_runbook.md) for prerequisites and explicit live actions.
