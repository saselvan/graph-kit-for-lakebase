# Lakebase Graph Kit

Turns your gold tables into a graph that Lakebase serves. A person guides each session, in the room with you.

## The graph

**Walk**:
One website question, expressed as one path through the graph with a maximum depth. Each depth is a prefix of that path.
_Avoid_: query, search, traversal

**Depth**:
The number of hops a walk takes. "Depth N" is the first N hops of the walk's path.
_Avoid_: level, degree

**Hop**:
One edge row in the served graph. A folded edge counts as one hop.
_Avoid_: step, join

**Folded edge**:
An edge that stands for a two-step path in the source data (for example condition ↔ condition through shared patients), stored as one weighted edge.
_Avoid_: derived edge, projected edge

**Enrichment**:
A score, group or link computed over the whole graph ahead of time, chosen by you from a fixed menu.
_Avoid_: feature, algorithm, analytics

## People and access

**Care subject**:
A patient or member whose data is protected. Their data is never served raw.
_Avoid_: person, user, patient record

**Public professional**:
A clinician or organisation from a public listing (for example a doctor with an NPI). Their data may be served.
_Avoid_: provider (alone), person

**Reader**:
The Postgres role the website uses. It can read only the served views or functions, never the tables under them.
_Avoid_: public role, app user

**Operator**:
The one Databricks identity that creates a graph's synced tables, owns its sync job and owns its served objects. In a workshop it is a person. For real use it is usually a service principal. A new operator means a new version, because access cannot be handed over.
_Avoid_: admin, owner, loader

## Sessions

**Run**:
One execution of the kit against one Lakebase project. A rerun is a new run.
_Avoid_: job, session, build

**Project**:
Your Lakebase project. Your admin makes it before the session. The kit never creates or changes one.
_Avoid_: database, instance, kit project

**Graph name**:
The short name the operator gives one graph, for example `providers`. Everything the kit owns in a project carries it.
_Avoid_: graph id, dataset name

**Kit schema**:
The one schema in a project that belongs to a graph. It carries the kit's marker. Its name defaults to `lgk_<graph>`, and the operator may choose another. It is the only place the kit writes.
_Avoid_: kit database, workspace

**Version**:
One complete build of a graph, held in its own version schema next to the others. The website reads exactly one version at a time.
_Avoid_: release, snapshot, rerun

**Served schema**:
The stable schema that the website reads. It holds only views or functions that point at the live version.
_Avoid_: public schema, API schema

**Switch**:
Pointing the served schema at a different version in one step. Rollback is a switch back.
_Avoid_: promote, deploy, cutover

**Refresh**:
The nightly run of the generated jobs that brings a version's data up to date. It does not change the graph's design and does not run the kit.
_Avoid_: rerun, rebuild, sync

**Manifest**:
The record of every object the kit created for a graph. Cleanup removes only what it lists.
_Avoid_: inventory, state

## Privacy

**Privacy floor**:
The smallest count the kit will serve. Smaller counts are hidden. The default is 11. You may raise it, never lower it.
_Avoid_: k, threshold, k-anon

## Test markers

**Seam-1**:
Live Lakebase integration test (pytest `seam1` marker; deselected by default, include with `--require-live`).

**Seam-2**:
Mock/no-cloud unit test (pytest `seam2` marker).
