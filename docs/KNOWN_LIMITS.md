# Known Limitations

## Sync Mode

The kit v1 supports only `snapshot` sync mode (the default). Any other `sync_mode` value (`"triggered"` or any unknown value) is refused with `UnsupportedSyncModeError` before any step builds anything.

## Folded Walks

Folded walks are web walks only. The `build` step folds every fold edge type and writes `edges.weight BIGINT` (the distinct care subjects behind a folded edge; NULL on unfolded edges). A folded tree walk raises `NotImplementedError`. A folded walk runs outbound only. A walk on a graph with folded and unfolded types must name its `edge_types` (a mixed selection is refused). Naming `edge_types` requires the run folder mapping (the mapping says which types are folded). The privacy step refuses a mapping that carries a dropped or folded column as a node or edge field.

## Enrichment Methods

The kit computes PageRank, degree, components, similarity and closure. PageRank has the following limits: **Maximum iterations**: 50 iterations, **Convergence tolerance**: L1 change below 1e-6, **Distance from converged answer**: up to 0.85/0.15 times the final L1 change. On the 1,000,000-node, 1,519,226-edge test graph, PageRank stopped at the 50-iteration cap without converging: the L1 change in the last iteration was 2.74e-6, so each score is within 1.55e-5 of the converged answer.

## Scale

Measured on a 1,000,000 nodes, 1,519,226 edges graph at 4 CU.

## Reference

Measure's reference reads node IDs and edges into the client and walks them in Python, without the graph's Lakebase views. It was checked on a 1,000,000 nodes, 1,519,226 edges graph, not beyond.

## Other Limitations

1. **The app is generated, never deployed**: the `app` step writes the app folder and prints the deploy command.
2. **Notebook runner config is JSON-only**: `NotebookRunner` refuses non-JSON config values (e.g., a `workspace_client` object).
3. **Notebook environment**: jobs run on serverless environment version 3 (Python 3.12); the notebook stops with a fix message on an older Python.
4. **Names are lower case only**: the mapping rejects a mixed-case name with a message giving the lower-case spelling.
5. **No default warehouse**: every UC statement requires `config["warehouse_id"]`.
6. **Build keeps a single node per natural key**: a duplicate key in the source keeps the row with the smallest (label, properties...) by string order, NULLs last.
