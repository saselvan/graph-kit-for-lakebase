"""The nightly refresh: a generated Lakeflow job, not a rerun of the kit.

- `job`: the job spec (SDP pipeline -> enrich -> sync -> reapply) and its creation.
- `modes`: the per-table sync mode and the triggered-needs-a-streaming-table check.
- `sync`: the sync task (start each synced table's pipeline update and wait).
- `freshness`: the guard for synced Delta version == source Delta version.
- `tasks`: the entry the job's notebook calls for each task.
"""
