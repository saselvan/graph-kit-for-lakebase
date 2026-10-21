"""How node scores are ranked: the one place that defines tie handling.

Scores are compared at SCORE_TIE_DECIMALS decimals, so float noise from Spark's
summation order cannot split a true tie. Ties are broken by (node_type, natural_key)
ascending: the natural key is what a person reads, and it does not depend on the hash.
"""

from typing import Any, Callable, List

SCORE_TIE_DECIMALS = 9
RANK_ORDER = f"ROUND(s.score, {SCORE_TIE_DECIMALS}) DESC, m.node_type ASC, m.natural_key ASC"


def top_nodes(sql: Callable[[str], List[List[Any]]], fq: str, method: str, limit: int) -> List[List[Any]]:
    """The top `limit` nodes for `method` as [node_type, natural_key, score] rows."""
    return sql(
        f"SELECT m.node_type, m.natural_key, s.score FROM {fq}.node_score s "
        f"JOIN {fq}.node_id_map m ON m.node_id = s.node_id "
        f"WHERE s.method = '{method}' ORDER BY {RANK_ORDER} LIMIT {int(limit)}"
    )
