"""The exported bundle's service principal: its own kit schema and its grants, as ONE list.

The bundle's SP builds and owns its own kit tables. The nightly
`build` CREATE OR REPLACEs nodes, edges and node_id_map, which needs ownership or MANAGE on the
table (MODIFY is not enough: live 2026-10-04, `PERMISSION_DENIED: User does not have MODIFY on
Table ...nodes`; https://docs.databricks.com/aws/en/data-governance/unity-catalog/access-control/privileges-reference).
So the bundle points at a kit schema of the SP's own (`<kit>_sp` by default), and every table
the SP rebuilds is one it created. The workshop's kit tables stay the operator's; the bundle
never reads or writes them.

The operator creates the SP's schemas (the SP cannot create schemas) and makes these grants.
The kit itself never grants anything on a source schema; it prints the statements.
Read by: the bundle README (lgk.export.operator_setup), the SP's preflight
(lgk.steps.preflight, `sp_builds`), the SKILL binding (tests/test_skill_freeze.py
check_sp_grants) and the live test's grants (tests/test_export_live.py).
"""

import re
from typing import Any, Dict, List, Tuple

SP_KIT_SUFFIX = "_sp"
SP_PLACEHOLDER = "<SP application id>"
PROJECT_PERMISSION = "CAN_USE"  # on the Lakebase project: the bundle never creates branches
PG_GRANT = "GRANT CONNECT, CREATE ON DATABASE databricks_postgres"

# (securable, target, privileges, operator creates it), in the order the operator runs them.
# target names a key of targets(): the full name is resolved from the bundle config + mapping.
UC_GRANTS: Tuple[Tuple[str, str, Tuple[str, ...], bool], ...] = (
    ("CATALOG", "lakebase_catalog", ("USE CATALOG",), False),
    ("CATALOG", "source_catalog", ("USE CATALOG",), False),
    ("SCHEMA", "sp_kit_schema", ("USE SCHEMA", "CREATE TABLE"), True),
    ("SCHEMA", "version_schema", ("USE SCHEMA", "CREATE TABLE"), True),
    ("SCHEMA", "source_schema", ("USE SCHEMA", "SELECT"), False),
)

# The SKILL's rendering: each target as a placeholder the operator fills in.
PLACEHOLDERS = {
    "lakebase_catalog": "<lakebase_catalog>",
    "source_catalog": "<source_catalog>",
    "sp_kit_schema": "<lakebase_catalog>.<sp_kit_schema>",
    "version_schema": "<lakebase_catalog>.<sp_kit_schema>_v<N>",
    "source_schema": "<source_catalog>.<source_schema>",
}


class WorkshopKitSchemaError(ValueError):
    """A bundle config whose kit schema is (or could be) the workshop's: the SP must build its own."""


def default_sp_kit_schema(workshop_kit: str) -> str:
    return f"{workshop_kit}{SP_KIT_SUFFIX}"


def require_own_kit_schema(config: Dict[str, Any]) -> str:
    """The bundle config's kit schema, refused unless it is the SP's own (never the workshop's).

    `workshop_kit_schema` (written by lgk.export) must be present and non-empty: a bundle
    without it was exported before the service principal's kit schema feature and would build
    into the operator's tables. The kit schema must be present, differ from it, and not be one of
    its version schemas.
    """
    workshop, kit = config.get("workshop_kit_schema"), config.get("kit_schema")
    if not workshop or not isinstance(workshop, str):
        raise WorkshopKitSchemaError(
            "the bundle config names no workshop_kit_schema: it was exported before the service "
            "principal built its own kit schema. Export the run again.")
    if not kit or not isinstance(kit, str):
        raise WorkshopKitSchemaError("the bundle config names no kit schema: the service principal builds "
                                     "only into its own, never a default")
    from lgk.versions import _checked_name

    try:  # plain lower-case identifiers, never a version schema name (`<x>_v<N>`, any digits)
        _checked_name(workshop, "workshop_kit_schema")
        _checked_name(kit, "kit_schema")
    except ValueError as e:
        raise WorkshopKitSchemaError(f"the bundle's kit schema names are not plain identifiers: {e}") from e
    # Unity Catalog names are case-insensitive: compare folded, so `K` is the workshop's `k`
    if kit.casefold() == workshop.casefold() or re.fullmatch(rf"{re.escape(workshop)}_v\d+", kit, re.I):
        raise WorkshopKitSchemaError(
            f"kit schema {kit!r} is the workshop's ({workshop!r}): the service principal builds only "
            "into its own kit schema; the workshop's tables are the operator's")
    return kit


def targets(config: Dict[str, Any], mapping: Any) -> Dict[str, str]:
    """Full UC names of each target, for the bundle (SP) config and the run's mapping."""
    from lgk.versions import version_schema

    cat = config["lakebase_catalog"]
    return {
        "lakebase_catalog": cat,
        "source_catalog": mapping.source_catalog,
        "sp_kit_schema": f"{cat}.{config['kit_schema']}",
        "version_schema": f"{cat}.{version_schema(config)}",
        "source_schema": f"{mapping.source_catalog}.{mapping.source_schema}",
    }


def uc_grants(names: Dict[str, str]) -> List[Tuple[str, str, Tuple[str, ...], bool]]:
    """(securable, full name, privileges, operator creates it), in UC_GRANTS order; a securable
    named twice (the source catalog is the Lakebase catalog) is granted once."""
    out, seen = [], set()
    for securable, target, privileges, creates in UC_GRANTS:
        key = (securable, names[target])
        if key in seen:
            continue
        seen.add(key)
        out.append((securable, names[target], privileges, creates))
    return out


def grant_sql(securable: str, name: str, privileges: Tuple[str, ...], principal: str) -> str:
    return f"GRANT {', '.join(privileges)} ON {securable} {name} TO `{principal}`;"


def create_schema_sql(name: str, graph_name: str) -> str:
    """The operator's CREATE SCHEMA, with the kit's ownership marker (lgk.manifest.marker)."""
    from lgk.manifest import marker

    comment = f"lgk:graph={graph_name}" if graph_name.startswith("<") else marker(graph_name)  # a placeholder
    return f"CREATE SCHEMA IF NOT EXISTS {name} COMMENT '{comment}';"


def setup_sql(names: Dict[str, str], graph_name: str, principal: str) -> List[str]:
    """The operator's Unity Catalog setup: create the SP's schemas, then every grant."""
    grants = uc_grants(names)
    return ([create_schema_sql(name, graph_name) for _, name, _, creates in grants if creates]
            + [grant_sql(securable, name, privileges, principal) for securable, name, privileges, _ in grants])
