"""The app generator: a deployable app folder plus one printed deploy command.

The generated app logs in as its own Databricks identity, becomes the Postgres role `reader`
on every connection, and serves ONE web walk through the served views only. The walk SQL is
the kit's (`lgk.steps.walks.get_walk_query`), with the privacy floor and folded flag taken
from the run's privacy gate, baked into the folder as text: the app does not need the kit.

The kit never deploys. It builds the `databricks apps deploy` command for the operator to run
with their own permissions, and calls no Apps API and runs no shell command.
"""

import re
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional

from lgk import versions
from lgk.privacy.gate import read_run_mapping, served_walk_selection
from lgk.served_functions import web_set_keys
from lgk.steps.walks import get_walk_query
from lgk.walks.sql import MAX_SEEDS_DEFAULT, NodePredicate

TEMPLATE_DIR = Path(__file__).parent / "template"
TEMPLATE_SUFFIX = ".tmpl"
# Templates with $placeholders; every other template is copied as is.
RENDERED = ("app.yaml", "backend/app/walk.py", "README.md", "DATA_CONTRACT.md")
CURSOR_RULE = ".cursor/rules/data-contract.mdc"
MARKER = ".lgk-app"

DEFAULT_DEPTH = 2
DEFAULT_TOP_N = 25
DEFAULT_DIRECTION = "outbound"
DEFAULT_ENDPOINT = "primary"
# Databricks Apps names: lower-case letters, digits and hyphens, 2..30 characters.
_APP_NAME = re.compile(r"[a-z][a-z0-9-]{0,28}[a-z0-9]")


class AppFolderError(RuntimeError):
    """The output folder holds files the kit did not generate; it is never overwritten."""


# ---------------------------------------------------------------------------- settings
def app_name(config: Dict[str, Any]) -> str:
    """config["app"]["name"], else `lgk-<graph>` with hyphens; must be a valid Apps name."""
    name = _app(config).get("name") or f"lgk-{config['graph_name']}".replace("_", "-")
    if not isinstance(name, str) or not _APP_NAME.fullmatch(name):
        raise ValueError(f"app name must match {_APP_NAME.pattern} (an Apps name), got {name!r}")
    return name


def endpoint_name(config: Dict[str, Any]) -> str:
    """projects/<p>/branches/<b>/endpoints/<e> of the branch the app reads."""
    from lgk.branch import BranchRef

    endpoint = _app(config).get("endpoint_id") or DEFAULT_ENDPOINT
    return f"{BranchRef.from_config(config).path}/endpoints/{endpoint}"


def _app(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("app") or {}


def _endpoint_host(config: Dict[str, Any]) -> str:
    """PGHOST: config["app"]["pghost"], else read once from the Lakebase endpoint (not Apps)."""
    host = _app(config).get("pghost")
    if host:
        return host
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient().postgres.get_endpoint(name=endpoint_name(config)).status.hosts.host


def _workspace_path(config: Dict[str, Any], name: str) -> str:
    """Where the operator copies the folder: config, else under the operator's home."""
    path = _app(config).get("workspace_path")
    if path:
        return path
    operator = config.get("operator_email")
    if not operator:
        from databricks.sdk import WorkspaceClient

        operator = WorkspaceClient().current_user.me().user_name
    return f"/Workspace/Users/{operator}/lgk-apps/{name}"


def deploy_command(name: str, workspace_path: str) -> str:
    """The one command the operator runs to deploy the app."""
    return f"databricks apps deploy {name} --source-code-path {workspace_path}"


# ---------------------------------------------------------------------------- the walk
def walk_queries(config: Dict[str, Any]) -> Dict[str, Any]:
    """The app's SQL (walk, start lookup, labels), all over the served schema only."""
    app = _app(config)
    served = versions.served_schema(config)
    depth = app.get("depth", DEFAULT_DEPTH)
    top_n = app.get("top_n", DEFAULT_TOP_N)
    direction = app.get("direction", DEFAULT_DIRECTION)
    # The edge types the app walks (required on a graph with folded and unfolded types);
    # folded is derived from them, so the app's support has one meaning
    floor, folded, edge_types = served_walk_selection(config["run_folder"], app.get("privacy_floor"),
                                                      None, app.get("edge_types"))
    walk = get_walk_query("", depth=depth, served_schema=served, privacy_floor=floor,
                          top_n=top_n, run_folder=config["run_folder"], direction=direction,
                          edge_types=edge_types)
    set_walks = _set_walk_queries(config["run_folder"], depth=depth, served_schema=served,
                                  privacy_floor=floor, top_n=top_n, edge_types=edge_types)
    return {
        "served_schema": served, "depth": depth, "top_n": top_n, "direction": direction,
        "privacy_floor": floor, "folded": folded, "edge_types": edge_types, "walk_sql": walk,
        "set_walk_sql": set_walks,
        "resolve_sql": (
            f"SELECT node_id, node_type FROM {served}.node_id_map_view "
            "WHERE natural_key = %(natural_key)s "
            "AND (CAST(%(node_type)s AS text) IS NULL OR node_type = %(node_type)s) "
            "ORDER BY node_type, node_id"
        ),
        "labels_sql": (
            f"SELECT node_id, node_type, natural_key, label FROM {served}.nodes_view "
            "WHERE node_id = ANY(%(node_ids)s)"
        ),
    }


def _key_predicate(key: str) -> NodePredicate:
    node_type, field = key.split(".", 1)
    return NodePredicate(node_type, field)


def _set_walk_queries(run_folder: str, **walk: Any) -> Dict[Any, str]:
    """{(seed key, stop key or None): set-walk SQL} over the served views, for every
    declared walk key ('<node_type>.<key>'); {} when the mapping declares none. The same gate
    selection as the app's walk (floor, edge types); outbound, at most MAX_SEEDS_DEFAULT seeds."""
    keys = web_set_keys(read_run_mapping(run_folder))
    return {(seed, stop): get_walk_query("", run_folder=run_folder, seed=_key_predicate(seed),
                                         stop=None if stop is None else _key_predicate(stop),
                                         max_seeds=MAX_SEEDS_DEFAULT, **walk)
            for seed in keys for stop in [None, *keys]}


# ---------------------------------------------------------------------------- rendering
# (what the walk direction means, the edges_view column a row's node_id comes from)
_DIRECTION_TEXT = {
    "outbound": ("along each edge, from src_node_id to dst_node_id", "dst_node_id"),
    "inbound": ("against each edge, from dst_node_id to src_node_id", "src_node_id"),
    "both": ("either way along each edge; a neighbour linked both ways counts once per edge type",
             "dst_node_id or src_node_id (the far end of the edge)"),
}


def _direction_text(q: Dict[str, Any]) -> Dict[str, str]:
    meaning, column = _DIRECTION_TEXT[q["direction"]]
    return {"direction_meaning": meaning, "node_id_column": column}


def _edge_types_text(q: Dict[str, Any]) -> Dict[str, str]:
    """Which edge types the walk follows: a closed list from the mapping, or every type."""
    types = q["edge_types"]
    return {"edge_types_text": ("every edge type of the graph" if types is None else
                                "only the edge types " + ", ".join(f"`{t}`" for t in types))}


def _privacy_text(q: Dict[str, Any]) -> Dict[str, str]:
    if q["folded"]:
        return {
            "support_meaning": "the number of distinct people (care subjects) behind the link "
                               f"from the previous hop; never below {q['privacy_floor']}",
            "privacy_rule": f"edges are folded care-subject data. A neighbour with fewer than "
                            f"{q['privacy_floor']} distinct people behind it is never served.",
        }
    return {
        "support_meaning": "the number of edges from the previous hop's nodes into this node "
                           "(a structural count; no people behind it)",
        "privacy_rule": f"the edges carry no care-subject data, so no count has people behind "
                        f"it; the floor ({q['privacy_floor']}) applies to folded edges only.",
    }


def _set_walk_text(q: Dict[str, Any]) -> Dict[str, str]:
    """The data contract's set-walk section (only when the graph declares walk keys)."""
    keys = sorted({k for k, _ in q["set_walk_sql"]})
    if not keys:
        return {"set_walk_contract": "", "other_walks_text": "no other walk"}
    listed = ", ".join(f"`{k}`" for k in keys)
    contract = (
        "### GET /api/set-walk?seed_key=<key>&seed_value=<value>&stop_key=<key, optional>"
        "&stop_value=<value, optional>\n\n"
        f"The set walk: every node whose key equals `seed_value` is a seed (at most "
        f"{MAX_SEEDS_DEFAULT}, more is an error), then depth {q['depth']}, at most {q['top_n']} "
        "nodes per hop over all seeds together, edges followed outbound. A reached node whose "
        "`stop_key` equals `stop_value` is returned with `matched` true and is not walked past. "
        f"Keys (`<node_type>.<property>`): {listed}. Equality is exact (case and spaces count).\n\n"
        "| Field | Type | Source |\n|---|---|---|\n"
        "| `seed.key`, `seed.value` | string | the query parameters |\n"
        "| `stop` | object or null | `{key, value}` from the query parameters; null without stop_key |\n"
        "| `rows[].hop`, `rows[].node_id`, `rows[].node_type`, `rows[].natural_key`, "
        "`rows[].label`, `rows[].support` | as `GET /api/walk` | support counts over all seeds |\n"
        "| `rows[].matched` | boolean | the node matches the stop predicate |\n\n"
        "Errors: 400 when a key is not listed above, or `stop_key` and `stop_value` do not come together.\n\n"
    )
    return {"set_walk_contract": contract, "other_walks_text": "no other walk than these two"}


def _values(config: Dict[str, Any], name: str, workspace_path: str) -> Dict[str, Any]:
    q = walk_queries(config)
    return {
        "app_name": name, "graph_name": config["graph_name"],
        "lakebase_project": config["lakebase_project"],
        "served_schema": q["served_schema"], "served_schema_repr": repr(q["served_schema"]),
        "depth": q["depth"], "top_n": q["top_n"], "privacy_floor": q["privacy_floor"],
        "folded": q["folded"], "direction": q["direction"],
        "walk_sql_repr": repr(q["walk_sql"]), "resolve_sql_repr": repr(q["resolve_sql"]),
        "labels_sql_repr": repr(q["labels_sql"]),
        "set_walk_sql_repr": repr(q["set_walk_sql"]), "max_seeds": MAX_SEEDS_DEFAULT,
        "pghost": _endpoint_host(config), "pguser": _app(config).get("pguser", ""),
        "endpoint_name": endpoint_name(config),
        "workspace_path": workspace_path,
        "deploy_command": deploy_command(name, workspace_path),
        **_privacy_text(q),
        **_direction_text(q),
        **_edge_types_text(q),
        **_set_walk_text(q),
    }


def render_files(values: Dict[str, Any]) -> Dict[str, str]:
    """{relative path: content} of every file in the app folder."""
    files = {}
    for src in sorted(TEMPLATE_DIR.rglob(f"*{TEMPLATE_SUFFIX}")):
        rel = src.relative_to(TEMPLATE_DIR).as_posix()[: -len(TEMPLATE_SUFFIX)]
        text = src.read_text()
        files[rel] = Template(text).substitute(values) if rel in RENDERED else text
    contract = files["DATA_CONTRACT.md"]
    files[CURSOR_RULE] = (
        "---\ndescription: Data contract of the generated graph search app (fields that exist, "
        "fields that do not, interpretation rules)\n"
        'globs: ["static/**", "backend/**"]\nalwaysApply: false\n---\n\n' + contract
    )
    files[MARKER] = f"generated by lakebase-graph-kit for graph {values['graph_name']}\n"
    return files


def _write_folder(out: Path, files: Dict[str, str]) -> None:
    """Write the folder; refuse a non-empty folder the kit did not generate."""
    if out.exists() and any(out.iterdir()) and not (out / MARKER).exists():
        raise AppFolderError(
            f"{out} is not empty and was not generated by the kit (no {MARKER}); "
            f"choose another app.output_dir"
        )
    for rel, text in files.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def generate_app(config: Dict[str, Any]) -> Dict[str, str]:
    """Write the app folder; return {"app_folder", "deploy_command"}. Never deploys."""
    name = app_name(config)
    workspace_path = _workspace_path(config, name)
    values = _values(config, name, workspace_path)
    out_dir: Optional[str] = _app(config).get("output_dir")
    out = Path(out_dir) if out_dir else Path(config["run_folder"]) / "app" / name
    _write_folder(out, render_files(values))
    return {"app_folder": str(out), "deploy_command": values["deploy_command"]}
