"""The fixed enrichment table shapes (spec: "Enrichment tables in Lakebase")."""

from typing import Dict, List, Tuple

TABLE_SHAPES: Dict[str, List[Tuple[str, str]]] = {
    "node_score": [("node_id", "STRING"), ("method", "STRING"), ("score", "DOUBLE"),
                   ("computed_at", "TIMESTAMP")],
    # Written by personalized PageRank, which is "coming".
    "seed_score": [("seed_id", "STRING"), ("node_id", "STRING"), ("score", "DOUBLE")],
    "node_group": [("node_id", "STRING"), ("method", "STRING"), ("group_id", "STRING")],
    "similar_to": [("src_id", "STRING"), ("dst_id", "STRING"), ("score", "DOUBLE")],
    "closure": [("ancestor_id", "STRING"), ("descendant_id", "STRING"), ("depth", "INT")],
}


def create_table_sql(fq_schema: str, table: str) -> str:
    """CREATE OR REPLACE one enrichment table with its fixed columns, in order."""
    cols = ",\n    ".join(f"{name} {typ}" for name, typ in TABLE_SHAPES[table])
    return f"CREATE OR REPLACE TABLE {fq_schema}.{table} (\n    {cols}\n) USING DELTA"
