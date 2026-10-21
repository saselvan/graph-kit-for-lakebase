---
name: lakebase-graph-kit
description: Interview an operator and build a graph from their gold Delta tables, served by Lakebase Postgres, using the lgk toolkit. Runs the toolkit steps preflight, profile, propose, privacy, build, enrich, provision, advise, switch and measure in order through a runner, shows how to walk the served graph, writes the app and the walks notebook, sets up the nightly refresh job and the service-principal bundle export, and retires, tears down or reaps a graph. Use when someone wants to turn a Unity Catalog schema into a Lakebase-served graph, resume a half-finished lgk run, or clean one up, or wants to query an lgk graph from an app, a notebook or SQL. Trigger phrases include "Lakebase Graph Kit", "lakebase-graph-kit", "lgk", "serve a graph from Lakebase", "find-a-doctor search", "shows related providers", "see which doctors work together", "which doctors work together", "related providers graph".
---

# Lakebase Graph Kit interviewer

**Completion invariant.** Never say a step, or the whole run, is done unless its **Done when**
line is met and you have shown the evidence for it: the command or query you ran and its output,
or the file content. If you cannot show the evidence, say the step is NOT DONE and what is
missing. Always end with the [Completion](#completion) step at the end of this file.

**Evidence rule.** Make a claim about latency (speed, p50/p95, "fast", "sub-second"), degree,
edge direction, graph shape (tree, web, DAG) or a count only when you show, next to the claim,
the exact query you ran and its output, both verbatim. If you have not shown them, do not make
the claim: say it was not measured. This rule fails closed. For the served graph's walk
correctness and walk speed, the evidence is `measure_report.md`, shown verbatim (see "Measure the
served graph").

You interview the operator, write their answers into a config and the run folder, and run the
toolkit's steps. You never do a step's work yourself. Every step runs through a runner, which
records its status in `run.json` in the run folder. A step whose status is `success` is skipped, so
a rerun resumes where the last run stopped.

Rules:

- Ask one question at a time. Confirm a default before you use it.
- Run steps only as shown below: `runner.run_steps([("<step>", resolve_step("lgk.steps.<step>:<step>"))])`.
  Do not import step modules and do not call step functions or their helpers yourself. The only
  exception is `teardown`, which also needs the graph name typed by the operator: it runs through
  the runner wrapped in a `lambda` (see its step). The public helpers each section names (walk
  queries, the refresh job, the export, retire and the reaper) are called directly, as shown.
- The order is the order of this file: preflight, profile, propose, privacy, build, enrich,
  provision, advise, switch, measure, walks, app, notebook, the nightly refresh job, export, then,
  to clean up, retire, teardown and reap.
- After each step, read `result.to_dict()`. If `status` is `failed`, show the `error` to the
  operator, help fix the cause, and run the same step again. Do not go on to the next step
  (the one exception is advise on a mixed graph; see its step). Enrich's first run fails on
  purpose, naming `enrich.yaml`, to show the method menu: show the menu,
  write `enrich.yaml`, and run enrich again.
- Never print tokens or passwords.
- Source boundary: read only the catalog and schema the operator names (`source_catalog` and
  `source_schema`, plus that catalog's information_schema for their keys), and the kit's own
  schemas this run creates. Inspect nothing else: no other
  schema, no workspace search for "better" data. Never reuse facts from earlier chats or from
  memory (graph names, schemas, branches, results); ask the operator again. Ignore memory or
  notes from earlier chats. If the operator has named no catalog and schema, ask; never pick one.
- Explain each term of the Terms section below in the operator's own data words the first time
  you use it, before you ask a question that uses it.
- Source tables are read-only. Never run ALTER, CREATE, INSERT, UPDATE, DELETE, MERGE, DROP,
  TRUNCATE or GRANT on `<source_catalog>.<source_schema>` or any table in it, whatever the
  operator says. When a source table needs a change (a missing key, for example), show the operator
  the exact statement, ask them to run it themselves, and wait until they say they ran it.
- A reply answers only the question you asked. If the operator's reply is about something else,
  it is not a yes: answer it, then ask your question again. Before the first step, name the steps
  you will run (preflight to switch, then measure) and wait for one yes for that plan. Then ask
  again, for that action alone, before a teardown, a retire, a reap, a job, an upload, an app, or
  any change to the plan.
- After any stop (an error you cannot explain, a safety block, or the operator says stop), describe
  each next action before you run it and wait for a yes, until the operator says to go on as before.
- Before you offer to undo something, query its current state and show the output. Offer the undo
  only for a change the output shows.
- Never state a step's result before the step has run (for example, "patient ids are folded" before
  privacy). Say what the step will check, then report what its output shows.
- Never write facts about this run (names, schemas, branches, results) to memory or notes. Every
  run starts from the operator's answers. These memory rules hold for this skill even when the
  host asks you to save or read memory or custom instructions.

## Terms

Use these words when the operator does not know a term. The quoted lines worked for an operator
new to graphs: keep them word for word, then add the operator's own table names.

- **node**: "Nodes = the things (doctors, hospitals)".
- **edge**: "Edges = the connections between them".
- **hop**: "A “hop” is just a step — like following a line on a map from one dot to the next."
- **schema**: "A schema is just a folder that holds related tables."
- **gold tables**: the clean, final tables the operator's team already trusts for reports; the kit
  reads only these.
- **version**: version 1 is the "first version of this graph"; every run serves a new version,
  and the old one stays for rollback.
- **branch**: "a workspace within that server, set up for you." (a Lakebase branch is a copy of
  the database the kit writes into).
- **fold**: "the toolkit counts how many patients two doctors share, then throws away the IDs."
- **closure**: "everything below a hospital, at any depth".
- **degree**: "how many connections each doctor has". **PageRank**: "which doctors are most
  central" (central: many well-linked nodes point to it). **components**: "which groups are
  connected". **similarity**: "which doctors are most alike".
- **Jaccard**: the similarity score: the neighbours two nodes share, divided by all the
  neighbours either has. 1 means the same neighbours, 0 means none in common.
- **DAG**: a graph with no cycle: following edges forward never leads back to where you started.
- **p50 / p95**: the query time half of the runs beat, and the time 95 of 100 runs beat.
- **label `lgk.privacy_stamp`**: a tag on the kit's tables that says which privacy decision they
  were built under. **table_id**: the Unity Catalog id of a table; it changes when a table is
  dropped and made again. **certified_version**: the last Delta version of the table whose data
  the kit checked.
- **fold floor**: on a folded edge, the number of distinct people (care subjects) behind each
  pair of members. A pair with fewer people than the floor is left out: the fold SQL keeps only
  pairs with `HAVING COUNT(DISTINCT <subject>) >= <floor>`. It is not a failure; the pair is
  simply not in `edges`.
- **join floor**: on a node `aggregates` join with a `subject`, the number of distinct people
  behind each node in the joined table. Below the floor that node's join outputs are left blank
  (NULL). It is not a failure either.
- Both floors are the one config value `privacy_floor`: default 11. It can be raised and never
  lowered (a value below 11 is refused).

## Interview: the config

Ask for each value. Defaults are the toolkit's own. Write every value into the config, defaults
too: build, enrich and provision read `run_folder` and `kit_schema` from the config and do not
fill them in.

| Key | Question | Default |
|---|---|---|
| `graph_name` | Short name for the graph, for example `providers`. | none |
| `run_folder` | Where should the run folder live? | `./runs/<graph_name>` |
| `lakebase_project` | Which Lakebase project serves the graph? | none |
| `branch_id` | Which branch of that project? Always ask; offer no branch name. Never assume `production`. The toolkit has no default branch: it refuses a config with no `branch_id` (`MissingBranchError`), and preflight requires it, so always write the answer into the config. | none |
| `source_catalog` | Which Unity Catalog catalog holds the gold tables? | none |
| `source_schema` | Which schema holds the gold tables? | none |
| `lakebase_catalog` | Which Unity Catalog catalog holds the kit's tables? | none |
| `kit_schema` | Which schema holds the kit's tables (`nodes`, `edges`, `node_id_map`)? | `lgk_<graph_name>` |
| `served_schema` | Which Postgres schema holds the served views and functions the website reads? | the kit schema name (`kit_schema`, which itself defaults to `lgk_<graph_name>`) |
| `sync_owner_user` | Who owns the sync job? It must be the identity that runs preflight and provision (the operator): preflight is `blocked` and provision refuses (`SyncOwnerMismatch`) for any other identity. A service principal owns it only through the exported bundle. | none |
| `warehouse_id` | Which SQL warehouse runs profile and build? Always ask, before profile: the toolkit has no default warehouse. Build, enrich and provision refuse a config without it (`MissingWarehouseError`); profile without it does not read the tables and writes declared verdicts only. | none |
| `version` | Which version number does this run provision? Use a new one every run (an existing version is refused). | `1` |

### Install the toolkit

If you run inside a Databricks workspace (an assistant in the workspace, not a laptop), skip this
section and the runner sections below: go to "Running inside a Databricks workspace".

Every step imports the toolkit, so install it first. Ask where the kit folder is: the unzipped kit
zip, or the toolkit repo. It holds `dist/lgk-0.1.0-py3-none-any.whl`,
`notebooks/lgk_runner.py` and `notebooks/lgk_refresh_task.py`. Run every command of this skill
from the kit folder. If `dist/` has no wheel (only in the repo), build it there with
`uv build --wheel`. Install the wheel into a Python 3.12 or newer, and show that it and the
Databricks SDK import. The wheel brings the SDK it needs; the command below names it too:

```bash
python3 -m pip install dist/lgk-0.1.0-py3-none-any.whl "databricks-sdk>=0.81"
python3 -c "import lgk, databricks.sdk; print(lgk.__file__)"
```

**Done when** the second command prints a path that ends in `lgk/__init__.py`.

Then ask which runner to use:

- **Laptop runner** (default): steps run in this process.
- **Notebook runner**: steps run in a serverless workspace notebook as the operator. Its config
  must be plain JSON values, so do not put `workspace_client` in it. Preflight's live permission
  checks still run: in the notebook, preflight makes its own workspace client as the operator.

Tell the operator now: teardown runs through the laptop runner (in this process) whichever
runner they choose, because it is wrapped in a `lambda` (see its step).

### Laptop runner

```python
from databricks.sdk import WorkspaceClient
from lgk.runner import LaptopRunner
from lgk.notebook_runner import resolve_step

config = {
    "graph_name": "<graph_name>",
    "run_folder": "<run_folder>",
    "lakebase_project": "<lakebase_project>",
    "branch_id": "<branch_id>",
    "source_catalog": "<source_catalog>",
    "source_schema": "<source_schema>",
    "lakebase_catalog": "<lakebase_catalog>",
    "kit_schema": "<kit_schema>",
    "served_schema": "<served_schema>",
    "sync_owner_user": "<sync_owner_user>",
    "warehouse_id": "<warehouse_id>",
    "version": 1,
    "workspace_client": WorkspaceClient(),
}
runner = LaptopRunner(config)
```

### Notebook runner

Ask for a work folder of the form `/Workspace/Users/<operator>/lgk_v1_<6 hex chars>`; the
toolkit refuses any other path. The runner needs the toolkit wheel (`dist/lgk-0.1.0-py3-none-any.whl`)
and the workspace notebook `notebooks/lgk_runner.py`, both from the kit folder (Install the toolkit).
Inside a Databricks workspace, give no `notebook_source` (see "Running inside a Databricks workspace").

`backend.setup()` below uploads that wheel and the runner notebook into the work folder, and
every runner notebook run installs the wheel with `pip install` before it runs a step.

Environment pin: every notebook run the kit submits, and every notebook task of the jobs it
makes (the refresh job, the bundle's jobs), runs on serverless environment version `3` (Python 3.12). The
default environment has an older Python and cannot import the kit; the runner notebook and the
refresh task notebook stop on a Python older than 3.12. Before the steps run, each notebook run writes
`<run_folder>/runtime.json`: `python_version` (`sys.version`), `python_version_info`,
`environment_version` (the `DATABRICKS_RUNTIME_VERSION` the runtime sets), the requested
`environment_version_requested`, and `steps`. Read it when a notebook run fails on its runtime.

After
`backend.setup()` in the block below has run, show the work folder:

```bash
databricks workspace list <work folder>
```

**Done when** that list shows `lgk-0.1.0-py3-none-any.whl` and `lgk_runner` in the work folder.

```python
from pathlib import Path

from databricks.sdk import WorkspaceClient
from lgk.notebook_runner import DatabricksNotebookBackend, NotebookRunner, resolve_step

config.pop("workspace_client", None)
backend = DatabricksNotebookBackend(
    WorkspaceClient(),
    workdir="<work folder>",
    wheel=Path("dist/lgk-0.1.0-py3-none-any.whl"),
    notebook_source=Path("notebooks/lgk_runner.py"),
)
backend.setup()
runner = NotebookRunner(config, backend)
```

When the run is finished, call `backend.cleanup()` to delete the work folder. The notebook
runner runs only module-level step functions (`module:qualname`), so `teardown`, which runs
wrapped in a `lambda`, needs the laptop runner: build a `LaptopRunner(config)` over the same run
folder (the notebook runner copies the run folder back after every step).

### Running inside a Databricks workspace

When you run inside a Databricks workspace (an assistant in the workspace, not a laptop), there is
no kit folder and no fixed working folder. These rules replace the kit-folder ones above:

- Use absolute paths only. Ask the operator for the workspace path of the wheel, for example
  /Workspace/Users/<operator>/lgk/lgk-0.1.0-py3-none-any.whl. Set `run_folder` to an absolute
  /Workspace/Users/<operator>/... path too.
- Install first, before you write the config or make a runner: the restart below clears every
  Python variable. Install in a Python cell, not a shell cell (a shell install does not reach the
  Python cells), then restart Python, then show that both import. After the restart, write the
  config (Interview: the config, without `workspace_client`) and make the runner as shown below.

```text
%pip install <wheel path> "databricks-sdk>=0.81"
dbutils.library.restartPython()
```

- Use the notebook runner. The runner notebook ships inside the wheel, so give no
  `notebook_source`; never read a notebook from the workspace as a file:

```python
from pathlib import Path

from databricks.sdk import WorkspaceClient
from lgk.notebook_runner import DatabricksNotebookBackend, NotebookRunner

backend = DatabricksNotebookBackend(WorkspaceClient(), workdir="<work folder>", wheel=Path("<wheel path>"))
backend.setup()
runner = NotebookRunner(config, backend)
```

- The `databricks` CLI here cannot read files that Python wrote on this machine. For the nightly
  job's task notebook (Nightly refresh), upload the copy packaged in the wheel instead of the CLI
  import:

```python
from databricks.sdk import WorkspaceClient
from lgk.notebook_runner import upload_notebook

upload_notebook(WorkspaceClient(), "/Workspace/Users/<operator>/lgk/lgk_refresh_task", "lgk_refresh_task.py")
```

- The app step is not tested inside a workspace. Give the operator the app folder path and
  the printed deploy command, and tell them to deploy from a laptop.

## Step 1: `preflight`

Checks what the run needs, as the identity that runs it. Every check only reads: preflight
creates, changes and grants nothing. It checks:

- the identity is `sync_owner_user`;
- the Lakebase project exists, and the identity holds `CAN_USE` on it. It asks for `CAN_MANAGE`
  only when the config sets `creates_branches: true`. No step of this skill creates a branch, so
  leave that key out;
- the branch exists, and Postgres on it answers as the identity;
- the kit schema and the version schema in `lakebase_catalog`: each exists with USE SCHEMA and
  CREATE TABLE for the identity, or the catalog grants CREATE SCHEMA so the kit can create it;
- USE CATALOG on `lakebase_catalog`, for the synced tables;
- the roles `reader`, `public_reader` and `coordinator_reader` each exist on the branch, or the
  identity can create roles;
- the served schema exists with CREATE on it for the identity, or the identity has CREATE on the
  database, so that switch can create the schema.

It lists each missing permission with a fix.

One-time owner setup for a `CAN_USE` identity. Such an identity cannot create roles or a Postgres
schema. The project owner runs this SQL once, on the branch or its parent (roles are copied to
child branches), with the identity's id in place of `<id>`. For a person, write `'user'` in place
of `'service_principal'`. The fix of a missing Postgres check gives this same setup:

```sql
CREATE EXTENSION IF NOT EXISTS databricks_auth;
SELECT databricks_create_role('<id>', 'service_principal');
CREATE ROLE reader WITH LOGIN;
CREATE ROLE public_reader WITH LOGIN;
CREATE ROLE coordinator_reader WITH LOGIN;
GRANT CREATE ON DATABASE databricks_postgres TO "<id>";
```

The identity also needs the databricks-sql-access workspace entitlement to use the SQL warehouse
(an identity with only workspace-consume cannot use the Lakebase API). Preflight does not check
this entitlement: if the first warehouse statement says the API is disabled, ask a workspace admin
to add it.

```python
results = runner.run_steps([("preflight", resolve_step("lgk.steps.preflight:preflight"))])
print([r.to_dict() for r in results])
```

Read `<run_folder>/preflight_result.json`. If `status` is `blocked`, show each entry of
`missing_permissions` (`permission` and `fix`) and stop until the operator has fixed them.

**Done when** `preflight_result.json` is not `blocked` and the step's status is `success`.

## Step 2: `profile`

Profiles the tables of `<source_catalog>.<source_schema>` through the SQL warehouse. It reads
each table's primary key and foreign keys from the Unity Catalog constraints
(`CONSTRAINT ... PRIMARY KEY` / `FOREIGN KEY ... REFERENCES`). If a key is missing, first run
this query and show its output:

```sql
SELECT table_name, constraint_name, constraint_type
FROM <source_catalog>.information_schema.table_constraints
WHERE table_schema = '<source_schema>'
```

If the output shows the key, tell the operator the profile missed it and stop; do not change the
tables. If the key is not there, show the operator the statement to run themselves (source tables
are read-only for you), wait until they say they ran it, then run profile again:

```sql
ALTER TABLE <source_catalog>.<source_schema>.<table> ADD CONSTRAINT <table>_pk PRIMARY KEY (<key column>)
``` To profile only some tables, write their names to
`<run_folder>/gold_tables.json`, for example `{"tables": [{"name": "providers"}]}`; without
it, every table in the schema is profiled.

```python
results = runner.run_steps([("profile", resolve_step("lgk.steps.profile:profile"))])
print([r.to_dict() for r in results])
```

The step writes `<run_folder>/schema.json`: per table, its keys, foreign keys, row count and
columns with match rate, distinct ratio and null rate. It holds verdicts only, never sampled
values. Show the operator the verdicts.

Key checks. Unity Catalog does not enforce key constraints, so profile also counts, in the data:
per table, `duplicate_key_rows` and `null_key_rows` of its key; per foreign key, `orphan_rows`
(rows whose foreign key matches no row of the table it references). These counts are in
`schema.json`; profile makes them only when the config has `warehouse_id`.

- A duplicate key fails profile with `DuplicateKeyError`, after it writes `schema.json`. The error
  names each table and its count. Show it to the operator. The source tables are read-only for you:
  ask the operator to fix the data (or to map a view with one row per key), then run profile again.
- Orphans do not fail profile. A row with an orphan foreign key cannot become an edge, so the
  operator decides for each foreign key with orphans. Ask one question for each foreign key: drop
  these rows from the graph, or fix the data first?

```python
from lgk.notebook_runner import resolve_step

open_orphans = resolve_step("lgk.keys:open_orphans")
print(open_orphans("<run_folder>"))
resolve_orphans = resolve_step("lgk.keys:resolve_orphans")
resolve_orphans("<run_folder>", "<table>", "<foreign key column>", "accept", chooser="<operator email>",
                reason="<reason the operator types>")
```

`open_orphans` lists each foreign key whose orphans no decision covers, with its counts.
`resolve_orphans` records one decision in `<run_folder>/key_decisions.json`:

- `"accept"`: build leaves those rows out of `edges` and records how many. An accept covers only
  the count it was made on. If the orphans grow, build stops with `OrphanKeyError` until the
  operator decides again.
- `"fixed"`: the operator fixed the data. Build then requires zero orphans for that foreign key.

`key_decisions.json` is not `flags.json`: a key decision never changes the privacy gate. Key
messages and records hold counts and names only, never a key value.

**Done when** `schema.json` lists every table the operator expects (every table in `gold_tables.json`, if it exists), the operator has seen the verdicts, and `open_orphans` returns an empty list.

## Step 3: `propose`

Drafts the graph: a node type for each keyed table, an edge for each foreign key, and one edge
for each join table (a table with no key and exactly two foreign keys). It refuses an edge whose
source or target table has no key. A table whose key holds a care-subject (patient) column is
never drafted as a node type: it is drafted only as a join-table edge, or skipped with a warning.

```python
results = runner.run_steps([("propose", resolve_step("lgk.steps.propose:propose"))])
print([r.to_dict() for r in results])
```

Show `<run_folder>/graph_mapping.yaml`. If you mention it, say that the `shape` in
`<run_folder>/flags.json` (`tree` or `web`) is only a guess, made from the table-level draft before
build and before any folded edge exists. The measured shape is the first line of
`<run_folder>/enrich_menu.md`, which enrich measures on the built nodes and edges (whether there is
a cycle, and how deep the graph goes). Never call the graph a tree (or a DAG) from this guess. The proposal is already in the form `build` reads (the contract in
`lgk.mapping`). Walk the operator through it and edit the file in place: rename, keep or drop
node and edge types, and change labels and properties. When you rename a node type, rename it
in every edge's `src_type` and `dst_type` too. A node type can also take properties from other tables of
the source schema with `joins` (see the commented example in the template): a `columns` join
copies columns of a table with one row per key, an `aggregates` join adds `sum`, `min`, `max`,
`count` or `count_distinct` of one column. An aggregate over a table with patient data needs
`subject: <its folded care-subject column>`; its outputs are blank for any node with fewer
distinct patients than the privacy floor. Never sum a FLOAT or DOUBLE column. If the edited file breaks the contract, `build`
stops before it runs any SQL, with a `MappingError` that names the field; fix that field and
run `build` again.

**Done when** the operator has confirmed every node type and every edge type in `graph_mapping.yaml`.

## Step 4: `privacy`

The privacy gate. Every column is classed as care subject (patient data), public professional,
or none, by its name and by its sampled values; a column the kit is unsure of counts as care
subject. The step writes `<run_folder>/flags.json` and FAILS while any care-subject flag is
open, so the run stops before build.

```python
results = runner.run_steps([("privacy", resolve_step("lgk.steps.privacy:privacy"))])
print([r.to_dict() for r in results])
```

Before you show the privacy output, explain it in plain words: a flag is a column that may hold
patient data; the operator decides for each one. To fold is: "the toolkit counts how many
patients two doctors share, then throws away the IDs." A pair with fewer people than the fold
floor (default 11) is left out of the graph. Say this again each time you show privacy output.

Read the open flags in `flags.json` and take them with the operator ONE BY ONE. For each flag,
show the column, its class and `detected_by` (name, values or both), and offer only the
choices its rules allow. Read `fold_allowed` in `flags.json` before you mention folding at all;
never promise a fold before you have read it.

- `fold`: only when `fold_allowed` is true. The kit sets it when the column is a foreign-key
  column of its table, or one column of a composite key (a key of two or more columns) of its
  table. A composite key membership is enough: no foreign key and no patients table is needed.
  A column that is its table's whole single-column key is never foldable. The column then
  reaches Lakebase only as folded, weighted edges whose weight (distinct people) is at or above
  the privacy floor; pairs below it are left out.
- `drop`: the column never reaches the graph. For a care-subject property on a node, say why
  fold is not offered: folding needs a key that links two other things.
- `reclassify`: to `public_professional` or `none`, only with a reason the operator types
  (`resolve_flag(..., "reclassify", chooser=..., reason="<typed reason>", new_class="none")`).

```python
from lgk.notebook_runner import resolve_step

resolve_flag = resolve_step("lgk.steps.privacy:resolve_flag")
resolve_flag("<run_folder>", "<table>", "<column>", "drop", chooser="<operator email>")
```

`resolve_flag` only records a decision (choice, chooser, reason, time) in the run folder's
`flags.json`. It never alters the source tables: a `drop` drops nothing in Unity Catalog. "drop"
removes the column from the GRAPH mapping only; the source table is unchanged.

Remove every dropped column from `graph_mapping.yaml`. A folded column leaves `src`, `dst` and
`properties` too, and is named instead as `fold_subject` of the edge it folds (or as `subject` of
an aggregate join): `build` then writes that edge type as folded edges with `edges.weight`, the
distinct people behind each pair. A `fold_subject` with `src` equal to `dst` links members that
share people (co-membership, one node type); otherwise each (src, dst) pair is one edge. Then run
the privacy step again. It fails if the mapping still uses a dropped or folded column as a node or
edge field, if a `fold_subject` names a column whose flag did not choose `fold`, and if a flag
chose `fold` but no edge or join folds it. A node type's `walk_keys` (the properties a set walk
may test by equality, at most 4) must be listed under its `properties`; a join output is never a
walk key. The floor is `privacy_floor` in
the config: default 11, it can be raised and never lowered. It also fails if the mapping uses a
column that `schema.json` does not list (an unclassified column is never served).

Node types and patient data. The privacy step fails with `CareKeyNodeError`, and
names each node type it refuses, when:

- the node type's `natural_key` holds a care-subject column, or a foreign key that leads (directly or through
  a chain of foreign keys) to a care-subject column or to a table keyed on one;
- the node type's source table or view holds ANY care-subject column, whatever its key.

The rule reads the classifier's class of each column. Only a `reclassify` to `public_professional`
clears a column: use it for a doctor's own name on a directory table (`providers.full_name`), with
the reason typed by the operator. Ask the operator first: "Is `<table>.<column>` the provider's
public name, not a patient's?" Reclassify only on a yes. A `drop` or a `reclassify` to `none` does
not clear it. A reclassify clears only that one column of that one table, and a copy or a view of
the table without the patient key is refused too when it still holds another care-subject column.
Fold is the only way patient data enters the graph: use the table only as an edge source and fold the care column. Known limit
(tell the operator; the kit does not detect it): a table with one row per person and no
care-subject column passes the gate. Do not map such a table as a node type.

The privacy step runs again on every run, and `build` and `provision` check its result before
any SQL. Any change to `schema.json`, `graph_mapping.yaml` or `flags.json` after it passed
makes build and provision refuse until the privacy step passes again. Never edit these files to
get past the gate.

No going back to an older decision without the privacy step. The privacy step records each
decision it clears in `<run_folder>/privacy_decisions.json`, the newest last. Only the privacy
step writes this file. Every step that checks the gate (build, enrich, provision, switch,
rollback, the nightly tasks, the app and the walks) refuses (`OlderDecisionError`) a decision that
is not the newest one recorded. So when the operator restores an older copy of `flags.json` and
`graph_mapping.yaml`, every step refuses it, also when that older decision was the current one
before. A decision the privacy step did not record, and a missing or damaged
`privacy_decisions.json`, are refused too (`UnrecordedDecisionError`). To use an older decision
again, restore its files and run the privacy step again: it records that decision as the newest.
Known limit (tell the operator; the kit does not detect it): a restore of
`privacy_decisions.json` together with `flags.json` and `graph_mapping.yaml` makes the older
decision current again.
Keep the run folder in git, so that a restore shows in the diff.

**Done when** no flag in `flags.json` is open, each flag has a decision with a chooser, and privacy ran green after the last edit.

## Step 5: `build`

Builds the generic Delta tables `nodes`, `edges` (deduplicated) and `node_id_map` in
`<lakebase_catalog>.<kit_schema>` from `graph_mapping.yaml`. Each edge type with a
`fold_subject` becomes folded edges whose `weight` is the number of distinct people behind the
pair (`weight` is blank on unfolded edges); a pair below the fold floor is left out (the fold
SQL's `HAVING COUNT(DISTINCT ...) >= <floor>`), not a failure. Node `joins` are LEFT JOINs: a
`columns` join whose key is not unique fails with `JoinNotUniqueError`, and a floored aggregate
output on a node below the join floor is left blank (NULL), not a failure. After writing, build
reads its tables back and fails with `FoldFloorError` or `JoinFloorError` only if a value below
the floor is still there (a check that the filter worked; provision and the nightly sync run the
same check before they sync).
Only after every row is written does build stamp `nodes`, `edges` and `node_id_map` with the
current privacy decision (the stamp that provision and the nightly sync check), so a
build that failed half way leaves tables provision refuses.

Build checks the keys again, on the data as it is now, before it replaces any kit table (the
nightly job builds without profile). It fails with `DuplicateKeyError` on a duplicate natural key
of a node type. It fails with `OrphanKeyError` on orphans that no decision covers: a foreign key
with no decision, a `"fixed"` decision while orphans remain, or more orphans than an accept
covers. The error names each table, column and count. Then take the decision with the operator
(the key checks of the profile step) and run build again. With an accept, build leaves those rows out
of `edges` and records the count in `key_decisions.json`.

```python
results = runner.run_steps([("build", resolve_step("lgk.steps.build:build"))])
print([r.to_dict() for r in results])
```

**Done when** the step's status is `success` and `nodes`, `edges` and `node_id_map` exist in `<lakebase_catalog>.<kit_schema>`.

## Step 6: `enrich`

Computes the graph enrichments the operator ticks from a fixed menu (scores, connected groups,
shared-neighbour similarity, and on graphs with no cycle the transitive closure), as Delta tables
in `<lakebase_catalog>.<kit_schema>`. Run it between build and provision.

Before the first enrich run, say this to the operator, word for word, and only then run the step:
"Enrich stops once on purpose to show you its menu. The red error is expected; nothing is wrong."

**First run done when** you said that line before the run, the run stopped on `enrich.yaml`, and
you showed the menu.

```python
results = runner.run_steps([("enrich", resolve_step("lgk.steps.enrich:enrich"))])
print([r.to_dict() for r in results])
```

- Before you show the enrichment output (the menu or the report), explain the methods in plain
  words, again each time: degree is "how many connections each doctor has", PageRank "which
  doctors are most central", components "which groups are connected", similarity "which doctors
  are most alike" (scored by Jaccard: shared neighbours divided by all neighbours of either),
  closure "everything below a hospital, at any depth" (only on a DAG: a graph with no cycle).
- The first run measures the graph, writes `<run_folder>/enrich_menu.md` and stops: the step
  fails with a `FileNotFoundError` that names `enrich.yaml`. This is expected.
- Show the operator the menu. Each row says what the method answers, an example on their graph
  and the table it adds. Greyed-out rows say why (for example, closure on a graph with a cycle).
  The menu shows no time number. Do not give the operator a time for a method either.
- The operator ticks methods; save them as `methods: [...]` in `<run_folder>/enrich.yaml`, for
  example `methods: [degree, pagerank]`. Then run the same step again: it computes those methods
  and writes `<run_folder>/enrich_report.json`.
- A greyed-out or unknown method is refused with the menu's reason, before any table is written.
  Remove it from `enrich.yaml` and run the step again.
- While enrich has failed, do not go on to provision (the rule above: never the next step after a
  failure). An operator who wants no enrichment saves `methods: []`; the step then writes a
  report and adds no table.
- Like build, enrich runs only when the privacy gate is clear. It removes every enrichment table
  the ticked methods do not write (an earlier run's table is never served), and marks each table
  it writes with the current privacy decision.

**Done when** `enrich_report.json` has one entry for every method in `enrich.yaml` (and none for `methods: []`).

## Step 7: `provision`

Syncs the built tables into the Lakebase branch as a NEW version (raw synced tables land in
Postgres schema `<kit_schema>_v<version>`) and makes sure the `reader`, `public_reader` and
`coordinator_reader` login roles exist with no right on them. It also stages the served functions
in the new version schema and calls each once, so a version they do not fit fails here. It does not serve the new version yet: the website keeps reading the old one. The
step then proves it: it fails if `reader` has any privilege on a raw table (directly or through
PUBLIC), is a superuser, or is a member of any other role. A failed proof leaves the site as it was.
Use a new `version` number for every run (an existing version is refused).
The enrichment tables the last enrich run wrote are synced too: scores, connected groups and
shared-neighbour similarity. The closure table is synced only when the config sets
`serve_closure: true` (closure is opt-in for now). Provision refuses an enrichment table made
under another privacy decision: run enrich again. It also indexes every declared walk key in the
new version, so the set walk's seed scan reads an index.

Operator setup for the roles: provision creates a missing role with plain `CREATE ROLE`. If the
running identity may not create roles, it fails with `ReaderRoleMissingError` and the SQL to run.
Ask the operator to run it once on the branch (or its parent; roles are copied to child
branches), then run provision again:

    CREATE ROLE reader WITH LOGIN; CREATE ROLE public_reader WITH LOGIN; CREATE ROLE coordinator_reader WITH LOGIN;

Sync limits: a source table holds at most 20 synced tables, and every version syncs every table
again. Before it creates any synced table, provision counts the `<kit_schema>_v<N>` schemas in
Unity Catalog that already hold each table and fails with `SyncLimitError` when one more would
pass 20 minus `sync_per_source_headroom` (config, default 2). The platform's own refusal of the
21st is reported the same way. Then retire an old version (or reap what a deleted branch left)
and run provision again.

```python
results = runner.run_steps([("provision", resolve_step("lgk.steps.provision:provision"))])
print([r.to_dict() for r in results])
```

**Done when** the step's status is `success` (its `reader` proof passed) and the Postgres schema `<kit_schema>_v<version>` exists on the branch.

## Step 8: `advise`

The index advisor runs the website's walk queries on the new, not-yet-served version, adds the
indexes that remove a scan of a large table, and writes `advisor_report.json` and
`advisor_report.md` (p50/p95 before and after) to the run folder. It refuses the version the
website already reads. Optional config: `advisor_shapes` (default web and tree; add `web_set` to
measure the set walk), `advisor_directions` (default `[outbound]`), `advisor_max_depth` (4) and
`advisor_seed_node_id` (default: the node with the most out-edges). If this identity may not read
the version's raw tables, the report says `advisor skipped: <reason>` and the step still succeeds.
The advisor walks every edge type, so on a graph with folded and unfolded edge types it is refused
with a `ValueError` (`edge_types is required ...`); the same holds for the notebook step.
Advise fails on every graph that mixes folded and unfolded edge types: the step
fails before it connects, so do not run it again. Tell the operator no index advice was made,
then go on to switch. This is the one failed step you may go past.

Before you show the advisor output, explain it in plain words, again each time: an index is a
lookup list that lets a walk find rows without reading the whole table; p50 is the query time
half of the runs beat, p95 the time 95 of 100 runs beat. Quote the numbers only from
`advisor_report.md`, shown verbatim (the evidence rule).

```python
results = runner.run_steps([("advise", resolve_step("lgk.steps.advise:advise"))])
print([r.to_dict() for r in results])
```

**Done when** `advisor_report.json` exists in the run folder (it may say `advisor skipped: <reason>`), or, on a graph that mixes folded and unfolded edge types, you have shown the operator the step's `edge_types is required` error and said advise was skipped.

## Step 9: `switch`

The last step: every served view and served function in `<served_schema>` moves onto the new
version in one transaction (the website sees the old version or the new one, never a mix), and
`reader` can read the served views only. The two function roles read the graph only through the
served functions, never a table: `public_reader` may run the public directory lookups, and
`coordinator_reader` may run the privacy-floored walks. The served views include
`node_score_view`, `node_group_view`, `similar_to_view` and `closure_view` when the version holds
that enrichment; a switch to a version without one removes its view. The first run creates the served views here. It is refused unless the
version was provisioned under the current privacy decision. The old version stays for rollback (a
switch back to it: "Roll back to an earlier version" below).

```python
results = runner.run_steps([("switch", resolve_step("lgk.steps.switch:switch"))])
print([r.to_dict() for r in results])
```

**Done when**, as `reader`, `SELECT count(*) FROM <served_schema>.nodes_view` returns rows, and `pg_views` shows `nodes_view` reading from `<kit_schema>_v<version>`.

## Measure the served graph

Run this after switch. Measure only reads: the served graph, and (through `warehouse_id`) the
source tables for its reference. It adds no index and no grant, and it changes no branch and no
CU range. It writes `<run_folder>/measure.jsonl` (one line per metric) and
`<run_folder>/measure_report.md`. Both hold counts and timings only, never a node key, a label or
a value. A rerun replaces both files.

- C10 walk correctness: for each shape and depth, measure runs the served query (the walk, or
  the similar lookup) for `measure_seeds` seeds (default 20). It compares each answer with a
  reference that reads only the source tables, never the kit's tables. The report counts the
  seeds checked, equal and different.
- C13 walk speed: client p95 for each shape and depth, timed on a warm connection with prepared statements off; Postgres execution time (EXPLAIN ANALYZE) is one part of the means breakdown. `measure_samples` runs (default 200, at least 20) follow
  `measure_warmup` runs (default 3).
- C23 time to first served graph: the sum of the step times in `run.json`, preflight to switch.
  It excludes operator time (the time between steps). A step with no time is named in the report.
- C30b freshness: pending, with the reason "nightly refresh untested from inside a workspace at
  this build".

Optional config: `measure_shapes` (default web, tree and similar), `measure_max_depth` (default
4: depths 1 to 4), `measure_similar_top_n` (default 10) and `measure_reference_max_rows`
(default 5,000,000 source rows for the reference). The similar shape is checked only when enrich
computed similarity; else its line says `not_served`. On a graph that mixes folded and unfolded
edge types, the web shape says `not_checked`.

```python
results = runner.run_steps([("measure", resolve_step("lgk.steps.measure:measure"))])
print([r.to_dict() for r in results])
```

Show `measure_report.md` verbatim. Quote a count, a p50 or a p95 of the served graph only from
it (the evidence rule). If a seed answer is different, tell the operator: the served walk and the
source tables disagree for that shape and depth.

To measure again (for example, after a nightly refresh), first mark the step to run again with
`runner.invalidate_steps(["measure"], "measure again")`, then run the block above.

**Done when** `<run_folder>/measure.jsonl` exists and you have shown `measure_report.md` verbatim, with its C10 table: seeds checked, equal and different for each shape and depth.

## Step 10: `walks`

There is no runner step for walks yet. The toolkit gives the walk query for a start node. Ask the
operator for a start node id (read it from `node_id_map`) and the depth (1 to 4; each hop keeps the top 25
nodes, ties broken by node id).
Whether the hop runs over folded edges, and the floor, come from the run folder's `flags.json`:
the walk refuses unless the privacy gate is clear and fresh. Over folded edges the served count is the number of
distinct people behind a neighbour, and a neighbour below the privacy floor (default 11, never
lower) is not shown. Edges with no care-subject data have no people behind them; the floor
does not apply to them.
Ask which way to follow the edges: `direction="outbound"` (the default: from the start node along each
edge), `"inbound"` (the nodes that point to it, for example the providers that point to a hospital) or
`"both"` (either way; a neighbour linked both ways counts once per edge type). Folded edges are symmetric,
so they are walked outbound only; any other direction is refused.
Ask which edge types to follow (`edge_types=[...]`, names from `graph_mapping.yaml`; default every
type). Whether the walk is folded comes from that choice: only folded types gives people counts,
only unfolded types gives edge counts, and a mix of both is refused (`MixedFoldSelectionError`), so
on a graph with folded and unfolded edge types `edge_types` is required. Naming `edge_types` needs
`run_folder`.

```python
from lgk.notebook_runner import resolve_step

get_walk_query = resolve_step("lgk.steps.walks:get_walk_query")
sql = get_walk_query("<start node id>", depth=2, served_schema="<served_schema>", run_folder="<run_folder>",
                     direction="outbound")
print(sql)
```

Run the query against the branch as the `reader` role, with `start_node_id` as its parameter.
To sort each hop by an enrichment score, add `sort_method="degree"` (or `"pagerank"`, if enrich
computed it); the walk reads the served `node_score_view`.

A set walk starts from every node that matches a seed instead of one start node: "every provider
with specialty X, two hops out, stopping at hospitals". It needs `walk_keys` in the mapping for a
field other than `label` or `natural_key`. Seed and stop are each a `NodePredicate(node_type,
field)`; the values are bound at run time as `seed_value` and `stop_value`. At most `max_seeds`
seeds (default 200, at most 1,000; more is an error, never a cut), outbound only. A node that
matches the stop is returned with `matched` true and is not walked past; a stop needs a seed.
A set walk takes exactly one seed condition (one walk key equal to one value) and at most one stop
condition. It cannot combine seed conditions ("Cardiology AND Springfield"): say so before
you offer a set walk, and never promise one.

```python
NodePredicate = resolve_step("lgk.walks.sql:NodePredicate")
sql = get_walk_query("", depth=2, served_schema="<served_schema>", run_folder="<run_folder>",
                     seed=NodePredicate("<node type>", "<walk key>"),
                     stop=NodePredicate("<node type>", "<walk key>"))
print(sql)
```

Run it as `reader` with `seed_value` and `stop_value` as its parameters.

On a graph with no cycle, where enrich computed `closure` and the config set
`serve_closure: true`, two lookups read `closure_view`: the ancestors or the descendants of a
node, up to `max_depth` levels (1 to 100), the first `top_n` (default 25) by depth then node id.
Example question: "Which hospitals sit above this clinic, up to 3 levels?"

```python
get_lookup_query = resolve_step("lgk.steps.walks:get_lookup_query")
sql = get_lookup_query("ancestors", served_schema="<served_schema>", run_folder="<run_folder>", max_depth=3)
print(sql)
```

Use `"descendants"` for the nodes below. The lookup is exact: it is cut once by (depth, node id),
while the tree walk cuts each level, so below a hub the two can differ. On a graph with a cycle
there is no closure; the lookup is refused with that reason, and the tree walk is the answer.
Run it as `reader` with `start_node_id`, like the walk.

Two more lookups need only `components` or `similarity` in enrich (they run on cyclic graphs too):
`get_lookup_query("same_group", ...)` returns the other nodes of the node's connected group
(`top_n` default 25, by node id), and `get_lookup_query("similar_jaccard", ...)` returns the nodes
that share the most neighbours with it, with their Jaccard score (`top_n` default 10, at most the
20 that enrich keeps per node; ties by node id). Example question: "Which providers share the most
neighbours with Dr X?" A lookup whose method was not computed is refused with the reason.

**Done when** the operator has run each query they asked for as `reader` and seen its rows, or its refusal and the reason.

## Step 11: `app` (app developers and architects only)

Only offer this to an app developer or architect; data engineers get the notebook. The step
writes a search app folder (default `<run_folder>/app/<name>`) that serves the web walk
(`GET /api/walk`, default depth 2, top 25 per hop) through the served views, as the `reader`
role, with the privacy floor from the run's privacy gate. When the mapping declares `walk_keys`,
the app also serves the set walk (`GET /api/set-walk?seed_key=...&seed_value=...`, optional
`stop_key` and `stop_value`; keys are `<node_type>.<property>`, at most 200 seeds). It takes one
seed condition and one stop condition only. Describe the app's endpoints to the operator only from the
generated `DATA_CONTRACT.md`, after this step has run. Every walk
runs in its own transaction with a 5 s statement timeout (`SET LOCAL statement_timeout = '5s'`),
so a slow walk is cancelled, never left running. The step prints exactly one
`databricks apps deploy ...` command. The toolkit never deploys the app: give the operator the
printed command and the folder's README.md and DATA_CONTRACT.md.

Operator setup for the app (the README's one-time steps): create the app, give its service
principal a Lakebase login that can become `reader` and nothing else
(`databricks_create_role(...)`, `GRANT CONNECT`, `GRANT reader`), and copy the folder to the
workspace path. Optional config: `app.name`, `app.depth`, `app.top_n`, `app.direction`,
`app.edge_types` (required on a graph with folded and unfolded edge types), `app.privacy_floor`
(raise only), `app.workspace_path`, `app.output_dir`, `app.pguser`, `app.pghost`,
`app.endpoint_id`.

```python
results = runner.run_steps([("app", resolve_step("lgk.steps.app:app"))])
print([r.to_dict() for r in results])
```

The kit writes the app folder only on this machine; it never creates the workspace folder the
printed `--source-code-path` names (default `/Workspace/Users/<operator>/lgk-apps/<name>`, or
`app.workspace_path`). Before the deploy command, upload the app folder to that exact path and
show it:

```bash
databricks workspace import-dir <app folder> <workspace path from the printed command> --overwrite
databricks workspace list <workspace path from the printed command>
```

**Done when** the app folder holds `README.md` and `DATA_CONTRACT.md`, the workspace list of the printed `--source-code-path` shows the uploaded files (`serve.py` among them), and the operator has the printed `databricks apps deploy` command.

## Step 12: `notebook` (data engineers)

The step writes `<run_folder>/walks_notebook.py`, a Databricks notebook with every walk the
website can run (web, and tree when the edges are not folded) at every depth: the exact SQL over
the served views only, a live answer from the branch, the time of the run and the advisor's
p50/p95 when `advisor_report.json` exists. It holds no answers: each cell queries the branch
again for the seed in its `seed_node_id` widget. It is refused unless the privacy gate is clear.
Optional config: `top_n` (25), `notebook_max_depth` (4), `notebook_seed_node_id` (default: the
advisor's seed). It walks every edge type, so a graph with folded and unfolded edge types is
refused with a `ValueError` (`edge_types is required ...`). Import the file into the workspace and run it as the operator.

```python
results = runner.run_steps([("notebook", resolve_step("lgk.steps.notebook:notebook"))])
print([r.to_dict() for r in results])
```

**Done when** `<run_folder>/walks_notebook.py` exists and the operator has imported it into the workspace.

## Nightly refresh

A refresh is not a rerun: do not run the steps again each night. After a successful
`provision`, offer the operator a nightly job. The toolkit generates one Lakeflow job whose tasks
run in this order, each after the one before. By default (no SDP pipeline): `build` (the kit
rebuilds nodes and edges from the source tables, through the same privacy gate as a manual
build), `enrich` (only when the run enriched, i.e. its run folder has `enrich.yaml`), the sync of every synced table (it first refuses tables not stamped with the
current privacy decision), then `reapply` (it waits up to 120 s for each synced table to report
its synced version, fails the job if none is reported in that time or if a synced table has missed
a data-changing commit of its source, then re-applies indexes, views and grants). Before all of
this, `reapply` refuses (`StaleBuildError`) a version that was not provisioned under the current
privacy decision: it is the same check as `switch`, and it applies also before the first switch. If the operator has
an SDP pipeline that refreshes the sources, ask for its id (`sdp_pipeline_id`): it runs first,
and replaces the kit build unless the config also says `build_mode: "sql"` (then the pipeline
refreshes the sources and the kit build rebuilds the graph after it). `build_mode` takes only
`"sql"`. Every refresh chain must carry the label `lgk.privacy_stamp` on nodes, edges and
node_id_map, and it has ONE meaning on every chain: the JSON `{stamp, table_id, certified_version}`
(the current cleared privacy decision, the table's UC table_id, the last Delta version whose data
was checked), recorded the same in the run manifest. The kit build writes it. When the operator's
pipeline rebuilds the kit tables (`sdp_pipeline_id` without `build_mode: "sql"`), the nightly job
runs a `stamp` task after the pipeline: it checks the privacy gate, the privacy floors and an
allowlist (exactly the kit's columns, and only the `properties` keys the cleared mapping allows),
then writes the label. The sync refuses (`StaleBuildError`) a table whose label does not match the manifest, the
current decision, the current table_id, or that got new data after it was checked (OPTIMIZE and
other maintenance do not count). A bare stamp the pipeline sets itself is no longer accepted. Tell
the operator this before they choose `sdp_pipeline_id` without `build_mode: "sql"`: their pipeline
must write plain Delta tables with exactly the kit's columns. Every job task installs the toolkit wheel from a workspace path, so upload the wheel and `notebooks/lgk_refresh_task.py` first.

Sync mode: v1 supports only `"snapshot"` (leave `sync_mode` at its default `"snapshot"`). When `sync_mode: "triggered"` is set, it is refused (`UnsupportedSyncModeError`) before any step builds anything.

The sync runs as many updates at once as the branch has room for, up to `sync_max_parallel`
(default 2). Before each update it reads the branch's free connections (max connections, minus the
reserved ones, minus every connection open now, minus `app_connection_headroom`, default 20) and
budgets 18 connections plus a margin of 2 per update (when an earlier night saw more than 18 sync
connections, that high-water mark plus 2). It stops with the numbers only if not
even one update fits; tell the operator to lower `app_connection_headroom`, wait for the app to
release connections, or raise the compute size.

Upload the wheel and the task notebook from the kit folder (Install the toolkit), then show the
folder. Inside a Databricks workspace the wheel is already at its workspace path: skip these
commands, upload the task notebook with `upload_notebook` (see "Running inside a Databricks
workspace"), and show the folder with your host's own folder-listing tool:

```bash
databricks workspace mkdirs /Workspace/Users/<operator>/lgk
databricks workspace import /Workspace/Users/<operator>/lgk/lgk-0.1.0-py3-none-any.whl --file dist/lgk-0.1.0-py3-none-any.whl --format AUTO --overwrite
databricks workspace import /Workspace/Users/<operator>/lgk/lgk_refresh_task --file notebooks/lgk_refresh_task.py --language PYTHON --format SOURCE --overwrite
databricks workspace list /Workspace/Users/<operator>/lgk
```

The job carries the config as JSON, so pass it without `workspace_client`. The job's serverless
tasks cannot read a laptop folder, so when `run_folder` is not a `/Workspace/` path,
`create_refresh_job` first checks that the run folder holds `flags.json`, `schema.json`,
`graph_mapping.yaml`, `privacy_decisions.json` and `lgk_manifest.json` (else it refuses with `NightlyFolderError`, before
any job exists). It then copies those files, and `enrich.yaml`, `enrich_report.json` and
`privacy_bindings.json` when they exist, to a workspace folder and points every task at that
folder: `workspace_run_folder` in the config, default `<notebook folder>/runs/<graph_name>`
(here `/Workspace/Users/<operator>/lgk/runs/<graph_name>`). That copy is the run folder of
record for the nightly chain; to re-seed it from the laptop, create the job again.
`create_refresh_job` also copies `key_decisions.json` (the operator's key decisions, which the
nightly build reads) when it exists. When `schema.json` records orphans and `key_decisions.json` is
missing, it refuses the same way. A `/Workspace/` `run_folder` is used as it is. Each task first
upgrades `databricks-sdk` and restarts Python, then installs the toolkit wheel, then runs.

```python
from lgk.notebook_runner import resolve_step

create_refresh_job = resolve_step("lgk.refresh.job:create_refresh_job")
job_config = {k: v for k, v in config.items() if k != "workspace_client"}
job_id = create_refresh_job(job_config, notebook_path="/Workspace/Users/<operator>/lgk/lgk_refresh_task",
                            wheel_path="/Workspace/Users/<operator>/lgk/lgk-0.1.0-py3-none-any.whl")
print(job_id)
```

The job is created with no schedule. Ask the operator for the time and time zone of the nightly
run, set the schedule, and show it from the job's settings:

```bash
databricks jobs update --json '{"job_id": <job id>, "new_settings": {"schedule": {"quartz_cron_expression": "0 0 2 * * ?", "timezone_id": "UTC", "pause_status": "UNPAUSED"}}}'
databricks jobs get <job id> -o json
```

A job is made for one version: the `version` in the config when you create it. After a switch to a
new version, that job's `reapply` task refuses (`ServedVersionError`), so each night would fail.
After each switch to a new version, create the refresh job again with the new config, set the
same schedule on it, and delete the old job (`databricks jobs delete <old job id>`).

Tell the operator the job id; deleting the job is part of cleaning up the graph. Teardown deletes
the job, but it does NOT delete the job's workspace copy of the run folder yet. The run manifest
records that folder in the job's entry, as `metadata.run_folder`. Tell the operator to delete it
after teardown and to show that it is gone:

```bash
databricks workspace delete <metadata.run_folder of the job entry> --recursive
databricks workspace list <metadata.run_folder of the job entry>
```

**Done when** the workspace list shows `lgk-0.1.0-py3-none-any.whl` and `lgk_refresh_task`, the job exists with a schedule, and you have shown the job id and the schedule (`settings.schedule.quartz_cron_expression` and `timezone_id`) from the `databricks jobs get` output.

## Export: `lgk.export:export(config)` (a bundle that provisions as a service principal)

When the team wants the graph in its own git and CI, export the run as a Databricks Asset
Bundle. The bundle never reuses the workshop's synced tables or its kit tables: the service
principal builds and owns its own kit tables, in a kit schema of its own (`sp_kit_schema`; by
default the kit schema's name with _sp added), because the nightly build replaces those tables
and only their owner may do that. Its one-time job `lgk_provision` runs preflight, build,
provision, advise and switch as a service principal (`run_as`) into a NEW version
(`version` + 1), and its nightly job `lgk_refresh` runs the same chain as the refresh job above.
The service principal's build goes through the same privacy gate and stamp as the operator's,
and its provision refuses any table it did not build under the current privacy decision.
The workshop's kit tables stay the operator's: the bundle never reads or writes them, and the
operator may tear them down. A run whose nightly chain has no kit build (an operator pipeline
without `build_mode: sql`) is refused by the export.
The bundle runs provision, advise and switch in order and stops at the first failure, so on a
graph that mixes folded and unfolded edge types `lgk_provision` fails at advise and never
switches: do not export such a graph.
No identity is written into the bundle (an email address in any file fails the export); the
service principal is the bundle variable `sp_app_id`, given at deploy time from `LGK_SP_APP_ID`.
The kit never deploys the bundle: it prints the commands.

```python
export = resolve_step("lgk.export:export")
bundle_dir = export(config)
print(bundle_dir)
```

Operator setup for the service principal (the bundle's README.md, once, before
`databricks bundle deploy`). In Unity Catalog the operator creates the service principal's kit
schema and version schema (it cannot create schemas) and makes these grants, the same list the
service principal's preflight checks. The kit never grants anything on a source schema:

```sql
CREATE SCHEMA IF NOT EXISTS <lakebase_catalog>.<sp_kit_schema> COMMENT 'lgk:graph=<graph_name>';
CREATE SCHEMA IF NOT EXISTS <lakebase_catalog>.<sp_kit_schema>_v<N> COMMENT 'lgk:graph=<graph_name>';
GRANT USE CATALOG ON CATALOG <lakebase_catalog> TO `<SP application id>`;
GRANT USE CATALOG ON CATALOG <source_catalog> TO `<SP application id>`;
GRANT USE SCHEMA, CREATE TABLE ON SCHEMA <lakebase_catalog>.<sp_kit_schema> TO `<SP application id>`;
GRANT USE SCHEMA, CREATE TABLE ON SCHEMA <lakebase_catalog>.<sp_kit_schema>_v<N> TO `<SP application id>`;
GRANT USE SCHEMA, SELECT ON SCHEMA <source_catalog>.<source_schema> TO `<SP application id>`;
```

Then `CAN_USE` on the Lakebase project
(not CAN MANAGE: the bundle never creates branches); in Postgres on the branch (or its parent),
`databricks_create_role('<SP application id>', 'service_principal')`, `GRANT CONNECT, CREATE ON
DATABASE databricks_postgres`, and the three reader roles (the SQL of the provision step). The
service principal never creates roles. Then set `LGK_SP_APP_ID` and run the printed commands.

**Done when** `bundle_dir` exists and the operator has the printed commands.

## Serve a new version

A new version (for example, after the operator changed the mapping or the flags) uses the same
run folder, so the rollback can still find the old one. The runner skips a step that already
succeeded, so first mark the steps to run again. Set `version` in the config to the next version
number after the one served now. Then, with the laptop runner, run
`runner.invalidate_steps(["build", "enrich", "provision", "advise", "switch"], "new version")`.

Then run the `build`, `enrich`, `provision`, `advise` and `switch` steps again, each exactly as
shown in its own section, with the same `runner`.

Leave `enrich` out of the list, and do not run it, if the run did not enrich. Check each step's "Done when" as
before. Then, if a nightly refresh job exists, create it again for the new version (Nightly
refresh).

**Done when** each of those steps shows status `success` and `pg_views` shows `nodes_view` reading
from `<kit_schema>_v<version>` (the new version).

## Roll back to an earlier version: `lgk.steps.switch:rollback(config, version)`

A rollback is a switch back to an earlier version. The runner skips a step that already
succeeded, so call `rollback` directly, as below. It is a `switch` to the version you name.

First, if a nightly refresh job exists, pause it. The job was made for the version served now.
After a rollback, its `reapply` task refuses (`ServedVersionError`) and does not move the
served views back, so each night would fail. Use the job's own schedule values:

```bash
databricks jobs update --json '{"job_id": <job id>, "new_settings": {"schedule": {"quartz_cron_expression": "<quartz_cron_expression>", "timezone_id": "<timezone_id>", "pause_status": "PAUSED"}}}'
databricks jobs get <job id> -o json
```

Then find the earlier version. Each switch adds an entry to `<run_folder>/versions.json`; the
last entry is the version served now. Show the list and ask the operator which version to serve.
An entry before the last may be a version made under another privacy decision; the rollback
refuses it (below), so then choose an earlier one:

```python
served_history = resolve_step("lgk.steps.switch:served_history")
print(served_history(config))
```

Then switch back. Give the version number as the second argument; do not change `version` in
the config.

```python
rollback = resolve_step("lgk.steps.switch:rollback")
rollback(config, 1)
```

The rollback refuses (`StaleBuildError`) a version provisioned under a privacy decision other than
the current one (for example, a version made before the operator tightened a flag), and a version
this run did not provision. Then choose an earlier version, or build and provision a new one.

The version you switch back to keeps the rows it had: the nightly sync refreshes only the version
its job was made for. To refresh the graph again, provision and switch a new version, create the
refresh job again for it (Nightly refresh), and delete the paused job.

**Done when**, as `reader`, `SELECT count(*) FROM <served_schema>.nodes_view` returns rows,
`pg_views` shows `nodes_view` reading from `<kit_schema>_v<version>` (the version you switched
to), the last entry `served_history` prints is that version schema, and, if a job exists, the
`databricks jobs get` output shows `settings.schedule.pause_status` as PAUSED.

## Retire an old version: `lgk.steps.retire:retire(config, version, confirm_graph_name)`

After a switch, the old version stays for rollback ("Roll back to an earlier version" above).
To drop it, ask the operator for the version number and to type the graph name exactly. Retire refuses the version
the served views read now, and any schema that holds anything but this graph's version tables.
Every check runs before any delete. It deletes the version's recorded synced tables and their
pipelines, then the event-log table of each of those pipelines (`event_log_<pipeline id>` in the
version schema), then the Postgres version schema, then the Unity Catalog version schema. The
pipeline id is the one the run manifest recorded, else the `pipeline_id` field Unity Catalog
gives for the synced table; with neither, while the Unity Catalog version schema exists, retire
refuses (`UnknownPipelineError`) before it deletes anything. To drop several at once, use
`resolve_step("lgk.steps.retire:retire_versions")(config, [<versions>], <typed name>)`: if any is
live, nothing is dropped.

```python
retire = resolve_step("lgk.steps.retire:retire")
retire(config, 1, "<graph name, typed by the operator>")
```

**Done when** the call returned and the version schema is gone from Postgres and from Unity Catalog.

## Step 13: `teardown`

Whole-graph teardown drops exactly what the run manifest (`<run_folder>/lgk_manifest.json`)
recorded, in this order: the recorded jobs (the refresh job), then this graph's grants revoked
from the shared roles, the served views and functions, the synced tables (and their pipelines),
the event-log table of each of those pipelines, the Postgres tables and schemas, then (tests only)
any recorded branch, then the Unity Catalog tables and schemas. Every check runs before any delete: each job must
carry this graph's tag, each schema this graph's marker and only recorded objects, and nothing
outside may depend on them. A missing object counts as already gone, so a teardown that failed
part way can run again. An exported bundle records its own jobs and synced tables in the bundle's
workspace state folder (`${workspace.root_path}/state/run`), not in this run folder, so a teardown
of this run folder does not remove them; tear the bundle's graph down with that folder as
`run_folder`. The call is `lgk.steps.teardown:teardown(config, confirm_graph_name)`:
ask the operator to type the graph name, and run it through the laptop runner wrapped in a
`lambda`, so `run.json` records it.

Tell the operator before they type: teardown needs the graph name typed to confirm (a different
name is refused), and it removes only what this graph's manifest says it created. Use the exact call
in the block below: the runner `lambda` passes the typed name as the second argument; its
second parameter is `confirm_graph_name` (there is no `confirm_name`). The shared
roles `reader`, `public_reader` and `coordinator_reader` are left in place: other graphs on the
branch use them.

```python
teardown = resolve_step("lgk.steps.teardown:teardown")
results = runner.run_steps([("teardown", lambda cfg: teardown(cfg, "<graph name, typed by the operator>"))])
print([r.to_dict() for r in results])
```

**Done when** `run.json` shows `teardown` with status `success`.

## Reap what a deleted branch left: `lgk.reaper:reap(config, confirm_graph_name)`

Deleting a branch does not delete its synced tables in Unity Catalog or their pipelines, and a
refresh job bound to that branch would sync into nothing every night. The reaper reads the run
manifest only: `find_orphan_jobs` lists each recorded job whose branch is gone, `find_orphans`
each recorded synced table whose branch is gone and of which something remains, and `reap` deletes
the orphan jobs first, then each orphan's synced table, UC table and recorded pipeline. Reap does
not drop the pipelines' event-log tables. Show the
operator both lists, then ask them to type the graph name.

```python
find_orphan_jobs = resolve_step("lgk.reaper:find_orphan_jobs")
print(find_orphan_jobs(config))
find_orphans = resolve_step("lgk.reaper:find_orphans")
print(find_orphans(config))
reap = resolve_step("lgk.reaper:reap")
print(reap(config, "<graph name, typed by the operator>"))
```

**Done when** `find_orphan_jobs(config)` and `find_orphans(config)` both return empty.

## Completion

Always the last thing you do, at the end of a session or when the operator asks whether the run
is done. Print every **Done when** line of each step you ran in this session, word for word, each
with its evidence (the command or query and its output, verbatim, or the file content), in this
form:

```text
<step>: Done when <the line, verbatim>
Evidence: <the command or query, and its output, verbatim>
Status: DONE | NOT DONE (<what is missing>)
```

Mark any line whose evidence you have not shown as NOT DONE, and say what is missing. Say the run
is done only when every line printed is DONE.
