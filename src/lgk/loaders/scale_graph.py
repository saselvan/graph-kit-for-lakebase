"""A synthetic graph of providers, practices, locations, systems, markets and networks.

Builds a graph with providers, practices, locations, systems, markets and networks,
plus referral edges between providers and membership edges between providers and practices.
All counts, ratios, names and value lists (like specialty terms) are invented here.
The SQL uses only range() and hash() functions; no random seed or clock.
The graph is built with SQL only, never touching any source data.
Each table declares its columns and keys, so profile and propose can draft the generator mapping.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from lgk import ident
from lgk.uc_sql import execute_sql

COUNTS = {"provider": 800_000, "practice": 120_000, "location": 50_000, "system": 20_000,
          "market": 9_000, "network": 1_000}
N_REFERRALS = 250_000
SECOND_PRACTICE_PCT = 25

# Invented specialty terms for provider properties
_SPECIALTIES = ["cardiology", "orthopedics", "neurology", "oncology", "rheumatology",
                "gastroenterology", "dermatology", "ophthalmology", "urology", "otolaryngology"]

_SCHEMA_MARKER = "scale graph: invented, sql only"


class ScaleGraphError(Exception):
    """The build or drop is refused."""


def node_counts(scale: float = 1.0) -> Dict[str, int]:
    """Scaled node counts per kind. The default scale is 1.0 (1 million nodes)."""
    return {kind: round(count * scale) for kind, count in COUNTS.items()}


def _validate_scale(scale: Any) -> float:
    """Raise ScaleGraphError if scale is not a positive finite real number."""
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ScaleGraphError(f"scale must be a real number, got {type(scale).__name__}")
    if not isinstance(scale, float):
        scale = float(scale)
    if scale <= 0 or not (scale == scale and scale != float("inf")):  # NaN, inf, negative, zero
        raise ScaleGraphError(f"scale must be positive and finite, got {scale}")
    return scale


def build_sql(catalog: str, schema: str, scale: float = 1.0) -> List[str]:
    """SQL statements to build the scale graph (order matters).

    First: CREATE SCHEMA. Then: one CREATE TABLE per node and edge table, each with columns and
    key constraints declared, followed by INSERT INTO. Profile reads the schema metadata and propose
    drafts the generator mapping from the declared primary and foreign keys.
    """
    catalog = ident.check(catalog, "catalog")
    schema = ident.check(schema, "schema")
    scale = _validate_scale(scale)

    counts = node_counts(scale)
    fq = f"{catalog}.{schema}"
    statements = []

    # Create schema
    statements.append(f"CREATE SCHEMA {fq} COMMENT '{_SCHEMA_MARKER}'")

    # Create node tables (parents before children, declared with columns and keys)

    # networks: no FKs
    statements.append(f"""CREATE TABLE {fq}.networks (
  network_id BIGINT NOT NULL,
  network_name STRING,
  CONSTRAINT pk_networks PRIMARY KEY (network_id)
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.networks
SELECT i AS network_id,
       CONCAT('Network ', i) AS network_name
FROM (SELECT id AS i FROM range({counts['network']}))""")

    # markets: FK to networks
    statements.append(f"""CREATE TABLE {fq}.markets (
  market_id BIGINT NOT NULL,
  market_name STRING,
  network_id BIGINT,
  CONSTRAINT pk_markets PRIMARY KEY (market_id),
  CONSTRAINT fk_markets_networks FOREIGN KEY (network_id) REFERENCES {fq}.networks
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.markets
SELECT i AS market_id,
       CONCAT('Market ', i) AS market_name,
       CAST(PMOD(HASH(i, 1), {counts['network']}) AS BIGINT) AS network_id
FROM (SELECT id AS i FROM range({counts['market']}))""")

    # systems: FK to markets
    statements.append(f"""CREATE TABLE {fq}.systems (
  system_id BIGINT NOT NULL,
  system_name STRING,
  market_id BIGINT,
  CONSTRAINT pk_systems PRIMARY KEY (system_id),
  CONSTRAINT fk_systems_markets FOREIGN KEY (market_id) REFERENCES {fq}.markets
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.systems
SELECT i AS system_id,
       CONCAT('System ', i) AS system_name,
       CAST(PMOD(HASH(i, 1), {counts['market']}) AS BIGINT) AS market_id
FROM (SELECT id AS i FROM range({counts['system']}))""")

    # locations: no FKs
    statements.append(f"""CREATE TABLE {fq}.locations (
  location_id BIGINT NOT NULL,
  location_name STRING,
  CONSTRAINT pk_locations PRIMARY KEY (location_id)
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.locations
SELECT i AS location_id,
       CONCAT('Location ', i) AS location_name
FROM (SELECT id AS i FROM range({counts['location']}))""")

    # practices: FKs to systems and locations (constraint names ordered alphabetically for propose)
    statements.append(f"""CREATE TABLE {fq}.practices (
  practice_id BIGINT NOT NULL,
  practice_name STRING,
  location_id BIGINT,
  system_id BIGINT,
  CONSTRAINT pk_practices PRIMARY KEY (practice_id),
  CONSTRAINT fk_practices_locations FOREIGN KEY (location_id) REFERENCES {fq}.locations,
  CONSTRAINT fk_practices_systems FOREIGN KEY (system_id) REFERENCES {fq}.systems
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.practices
SELECT i AS practice_id,
       CONCAT('Practice ', i) AS practice_name,
       CAST(PMOD(HASH(i, 1), {counts['location']}) AS BIGINT) AS location_id,
       CAST(PMOD(HASH(i, 2), {counts['system']}) AS BIGINT) AS system_id
FROM (SELECT id AS i FROM range({counts['practice']}))""")

    # providers: no FKs
    specialties_sql = ", ".join(f"'{s}'" for s in _SPECIALTIES)
    statements.append(f"""CREATE TABLE {fq}.providers (
  provider_id BIGINT NOT NULL,
  provider_name STRING,
  specialty STRING,
  CONSTRAINT pk_providers PRIMARY KEY (provider_id)
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.providers
SELECT i AS provider_id,
       CONCAT('Provider ', i) AS provider_name,
       ELEMENT_AT(ARRAY({specialties_sql}), CAST(PMOD(HASH(i, 1), {len(_SPECIALTIES)}) AS INT) + 1) AS specialty
FROM (SELECT id AS i FROM range({counts['provider']}))""")

    # Create edge tables

    # memberships: join table with FKs to providers and practices
    # constraint names ordered: fk_a before fk_b so provider is src, practice is dst
    statements.append(f"""CREATE TABLE {fq}.memberships (
  provider_id BIGINT,
  practice_id BIGINT,
  CONSTRAINT fk_memberships_a_providers FOREIGN KEY (provider_id) REFERENCES {fq}.providers,
  CONSTRAINT fk_memberships_b_practices FOREIGN KEY (practice_id) REFERENCES {fq}.practices
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.memberships
SELECT provider_id,
       practice_id
FROM (
  SELECT provider_id,
         CAST(PMOD(HASH(provider_id, 1), {counts['practice']}) AS BIGINT) AS practice_id
  FROM (SELECT id AS provider_id FROM range({counts['provider']}))
  UNION ALL
  SELECT provider_id,
         CAST(PMOD(HASH(provider_id, 2), {counts['practice']}) AS BIGINT) AS practice_id
  FROM (SELECT id AS provider_id FROM range({counts['provider']})
        WHERE PMOD(HASH(id, 0), 100) < {SECOND_PRACTICE_PCT})
)""")

    # referrals: self-referencing edge with two FKs to providers
    statements.append(f"""CREATE TABLE {fq}.referrals (
  src_provider_id BIGINT,
  dst_provider_id BIGINT,
  CONSTRAINT fk_referrals_a_src_providers FOREIGN KEY (src_provider_id) REFERENCES {fq}.providers,
  CONSTRAINT fk_referrals_b_dst_providers FOREIGN KEY (dst_provider_id) REFERENCES {fq}.providers
) USING DELTA""")
    statements.append(f"""INSERT INTO {fq}.referrals
SELECT src_provider_id,
       dst_provider_id
FROM (
  SELECT id,
         CAST(PMOD(HASH(id, 1), {counts['provider']}) AS BIGINT) AS src_provider_id,
         CAST(PMOD(HASH(id, 2), {counts['provider']}) AS BIGINT) AS dst_provider_id
  FROM (SELECT id FROM range({round(N_REFERRALS * scale)}))
) WHERE src_provider_id != dst_provider_id""")

    return statements


def mapping_yaml(catalog: str, schema: str) -> str:
    """Graph mapping YAML for the scale graph."""
    catalog = ident.check(catalog, "catalog")
    schema = ident.check(schema, "schema")

    mapping = {
        "source_catalog": catalog,
        "source_schema": schema,
        "nodes": {
            "providers": {
                "source_table": "providers",
                "natural_key": "provider_id",
                "label": "provider_name",
                "properties": ["specialty"],
            },
            "practices": {
                "source_table": "practices",
                "natural_key": "practice_id",
                "label": "practice_name",
            },
            "locations": {
                "source_table": "locations",
                "natural_key": "location_id",
                "label": "location_name",
            },
            "systems": {
                "source_table": "systems",
                "natural_key": "system_id",
                "label": "system_name",
            },
            "markets": {
                "source_table": "markets",
                "natural_key": "market_id",
                "label": "market_name",
            },
            "networks": {
                "source_table": "networks",
                "natural_key": "network_id",
                "label": "network_name",
            },
        },
        "edges": {
            "memberships": {
                "source_table": "memberships",
                "src": "provider_id",
                "src_type": "providers",
                "dst": "practice_id",
                "dst_type": "practices",
            },
            "practices_system_id": {
                "source_table": "practices",
                "src": "practice_id",
                "src_type": "practices",
                "dst": "system_id",
                "dst_type": "systems",
            },
            "practices_location_id": {
                "source_table": "practices",
                "src": "practice_id",
                "src_type": "practices",
                "dst": "location_id",
                "dst_type": "locations",
            },
            "systems_market_id": {
                "source_table": "systems",
                "src": "system_id",
                "src_type": "systems",
                "dst": "market_id",
                "dst_type": "markets",
            },
            "markets_network_id": {
                "source_table": "markets",
                "src": "market_id",
                "src_type": "markets",
                "dst": "network_id",
                "dst_type": "networks",
            },
            "referrals": {
                "source_table": "referrals",
                "src": "src_provider_id",
                "src_type": "providers",
                "dst": "dst_provider_id",
                "dst_type": "providers",
            },
        },
    }
    return yaml.dump(mapping, default_flow_style=False, sort_keys=False)


def build_scale_graph(w: Any, warehouse_id: str, catalog: str, schema: str,
                      mapping_path: str, scale: float = 1.0) -> None:
    """Build the scale graph and write the mapping.

    Validates names and scale first; checks if schema already exists; runs SQL if clear;
    writes mapping last.
    """
    # Validate names and scale before any SQL
    catalog = ident.check(catalog, "catalog")
    schema = ident.check(schema, "schema")
    scale = _validate_scale(scale)

    fq = f"{catalog}.{schema}"

    # Check if schema already exists (before any CREATE or mapping write)
    lookup_stmt = f"SELECT comment FROM {catalog}.information_schema.schemata WHERE schema_name = '{schema}'"
    rows = execute_sql(w, warehouse_id, lookup_stmt)
    if rows:
        raise ScaleGraphError(f"schema {fq} already exists")

    # Run SQL statements
    for stmt in build_sql(catalog, schema, scale):
        execute_sql(w, warehouse_id, stmt)

    # Write mapping
    with open(mapping_path, "w") as f:
        f.write(mapping_yaml(catalog, schema))


def drop_scale_graph(w: Any, warehouse_id: str, catalog: str, schema: str) -> None:
    """Drop the scale graph schema if it exists and was created by the build.

    Raises ScaleGraphError if the schema does not exist or was not created by us.
    """
    catalog = ident.check(catalog, "catalog")
    schema = ident.check(schema, "schema")

    fq = f"{catalog}.{schema}"

    # Check if schema exists and has the marker comment
    lookup_stmt = f"SELECT comment FROM {catalog}.information_schema.schemata WHERE schema_name = '{schema}'"
    rows = execute_sql(w, warehouse_id, lookup_stmt)

    if not rows:
        raise ScaleGraphError(f"schema {fq} does not exist")

    if len(rows) != 1:
        raise ScaleGraphError(f"expected exactly one schema, found {len(rows)}")

    comment = rows[0][0] if rows[0] else None
    if comment != _SCHEMA_MARKER:
        raise ScaleGraphError(f"schema {fq} was not created by the scale graph builder")

    # Drop the schema
    execute_sql(w, warehouse_id, f"DROP SCHEMA {fq} CASCADE")
