# Lakebase Graph Kit

A deterministic Python toolkit for building graphs from Delta tables and serving them through Lakebase (Postgres).

## Big Idea

If your graph questions start from a known entity and go a few hops out, you don't need a separate graph database: build the graph from your Unity Catalog tables and serve it from Lakebase.

## Data Flow

![Data Flow Diagram](docs/readme-assets/data-flow.svg)

## Overview

The kit turns a Unity Catalog schema into a graph served by Lakebase by running 13 steps in order, each reading the previous step's artifacts and writing its own:

1. **preflight** — Confirms the environment and lists each missing permission with its fix
2. **profile** — Profiles the schema without sampling sensitive data
3. **propose** — Drafts the graph structure from keyed tables and foreign keys
4. **privacy** — The gate between the interview and build: column classification, then `flags.json`, one flag per care-subject column, resolved one by one
5. **build** — Creates the `nodes` and `edges` Delta tables from `graph_mapping.yaml`
6. **enrich** — Computes the enrichments the user ticked on the fixed menu (Spark batch)
7. **provision** — A new version in Lakebase, not yet served
8. **advise** — Runs the index advisor on the new version before the switch: proposes indexes and applies them
9. **switch** — Points the served views at one version, in one transaction; a rollback points them back at an earlier version
10. **walks** — The public query helpers over the served views
11. **app** — Generates the search app folder and prints its one deploy command
12. **notebook** — Writes the data-engineer walks notebook to the run folder
13. **teardown** — Drops exactly what the run manifest names

After switch, measure is run as its own command to record performance into `measure.jsonl` and `measure_report.md`.

## Prerequisites

The kit requires these platform features:

- [Lakebase projects and branches](https://docs.databricks.com/aws/en/oltp/projects)
- [Branch expiry](https://docs.databricks.com/aws/en/oltp/projects/manage-branches)
- [Synced tables](https://docs.databricks.com/aws/en/oltp/projects/sync-tables)
- [Postgres roles for the reader](https://docs.databricks.com/aws/en/oltp/projects/manage-roles)
- [SQL warehouses](https://docs.databricks.com/aws/en/compute/sql-warehouse)
- [Serverless notebooks (notebook runner and measure)](https://docs.databricks.com/aws/en/compute/serverless)
- [Databricks Apps (optional)](https://docs.databricks.com/aws/en/dev-tools/databricks-apps)
- [Unity Catalog](https://docs.databricks.com/aws/en/data-governance/unity-catalog)

## Installation

```bash
uv sync
```

## Quick start

```python
from databricks.sdk import WorkspaceClient
from lgk.runner import LaptopRunner
from lgk.notebook_runner import resolve_step
from lgk.loaders.mondo import load_mondo

config = {
    "run_folder": "./runs/my_graph",
    "lakebase_project": "<project>",
    "branch_id": "<branch>",
    "source_catalog": "<source_catalog>",
    "source_schema": "<source_schema>",
    "lakebase_catalog": "<lakebase_catalog>",
    "kit_schema": "lgk_my_graph",
    "served_schema": "lgk_my_graph",
    "sync_owner_user": "<operator_email>",
    "warehouse_id": "<warehouse>",
    "workspace_client": WorkspaceClient(),
}

# Optional starter data: writes three MONDO tables and a volume into source_schema. Use a new, empty schema, or skip this step to use your own tables.
load_mondo(config["workspace_client"], config["warehouse_id"],
           config["source_catalog"], config["source_schema"],
           config["run_folder"])

runner = LaptopRunner(config)

# Run the first few steps
steps = [
    ("preflight", resolve_step("lgk.steps.preflight:preflight")),
    ("profile", resolve_step("lgk.steps.profile:profile")),
    ("propose", resolve_step("lgk.steps.propose:propose")),
]

results = runner.run_steps(steps)
for result in results:
    print(result.to_dict())
```

## File Tree

```
run_folder/
├── run.json                      # Step status and timing
├── lgk_manifest.json            # Record of all created objects
├── gold_tables.json             # Input mapping (graph structure)
├── lg_mapping.yaml              # Processed graph mapping
├── measure.jsonl                # Performance metrics
└── measure_report.md            # Performance summary report
```

## Design

- **Runner seam**: Abstract interface for invoking steps, with laptop and notebook implementations
- **Run folder**: Stores `run.json` (per-step status), manifests, and artifacts
- **Manifest**: Records all created objects for safe teardown
- **Deterministic hashing**: Node IDs use hash-based keys for stable edges

## Testing

```bash
uv run pytest -v
```

Live tests are deselected by default and included with `--require-live`. They keep the Databricks CLI's default OS-keyring token storage. Under many
parallel live suites the keyring can fail. To avoid that, opt in with `LGK_LIVE_AUTH_STORAGE=plaintext`: the root `conftest.py` then sets
`DATABRICKS_AUTH_STORAGE=plaintext` for the test process and its children only (never if you set
`DATABRICKS_AUTH_STORAGE` yourself). The CLI then keeps tokens in `~/.databricks/token-cache.json`
(restricted permissions, outside the repo). Log the profile in once with that storage first:
`DATABRICKS_AUTH_STORAGE=plaintext databricks auth login --profile <profile>`.

Live tests can also record their numbers (walk and lookup latency percentiles).
This is opt-in and off by default: set `LGK_METRICS_FILE=<path>` and each
live test appends one JSON line per metric to that file (created if missing; nothing is written when
the variable is unset). A line holds `metric`, `value`, `unit`, `test` (pytest node id), `suite`,
`commit` (`git rev-parse HEAD` of the tree under test), `branch` (the `t-` branch), `project`,
`endpoint_cu` (min/max, read-only), optional `n_samples`/`depth`/`kind`/`direction`/`table`/`role`,
and `ts` (UTC). The recorder refuses a call (`MetricRefused`) when the value is not a finite number, the unit is not on its fixed list, or a context key is not on its fixed list; a context value must be a non-negative whole number or a short identifier-shaped name. A refused call raises and writes no line, so a line holds numbers and short names only, never free text.

Every line also says what it was measured at, and the recorder requires these fields:
- `rows` and `edges`: the size of the graph the test provisioned (its served nodes and edges views).
- `run`: one of `dogfood`, `load`, `advisor`, `refresh`, `test`, from `LGK_METRICS_RUN` (default
  `test`; any other value fails the test at setup).
- `client`: `laptop` (timed from the pytest process, so it includes the laptop-to-region network) or
  `workspace` (timed in-region). The walks and lookups suites each run one serverless notebook job on
  their own `t-` branch, through the shipped notebook runner and a wheel of the tree under test. The
  job runs web walks and lookups over the same seeds, times each query,
  and sends the samples back in the run folder. The test records their latencies as `client: workspace`.

Freshness lag (`refresh.freshness_lag`, unit `min`, one line per synced table, recorded by the
nightly tests of `test_refresh_live`) is defined as the sync completion time minus the source Delta
commit time of the data the synced table now holds. The completion time is when the synced table
reached ONLINE_NO_PENDING_UPDATE after the update: `status.last_sync.sync_end_time`. The commit time
is `status.last_sync.delta_table_sync_info.delta_commit_time`, cross-checked against DESCRIBE HISTORY
of the source at `delta_commit_version`. Never `last_sync_time`, which is always None. A reading from
before the night's update, or a sync that ends before its commit, is refused.

Summarise one run: `uv run --no-sync python tools/metrics_summary.py /tmp/lgk-metrics.jsonl [--run test]`.
Every row shows its run, client, rows and edges. The summariser refuses a file that mixes commits,
and a file that mixes runs unless `--run` picks one.

## Known Limitations

See [Known Limitations](docs/KNOWN_LIMITS.md) for current limitations on sync modes, folded walks, enrichment methods, and other features.

## Interviewer Skill

The interviewer is one committed skill folder, `skill/lakebase-graph-kit/` (`SKILL.md` with only
`name` and `description` frontmatter). It drives the toolkit only through the runner interface
(`LaptopRunner` / `NotebookRunner` + `resolve_step("lgk.steps.<step>:<step>")`). Install the same
bytes into both skill hosts:

```python
from pathlib import Path
from lgk.interview import install_skill_to_both_hosts

install_skill_to_both_hosts(Path.home() / ".claude", Path("<local copy of /Users/<me>>"))
```

The Genie Code copy must then be uploaded to `/Users/<me>/.assistant/skills/lakebase-graph-kit/SKILL.md`
in the workspace. `tests/test_skill.py` checks the frontmatter, that every toolkit name in the
skill resolves, that steps run only through `run_steps`, and that both installed copies equal the
committed file.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Provided for exploration; not formally supported by Databricks.
