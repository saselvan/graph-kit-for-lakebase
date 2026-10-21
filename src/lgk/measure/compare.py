"""Served answers vs the source reference (C10).

Comparison is order-insensitive (both sides are normalised to sorted tuples), and cells are
normalised by type (the driver may hand back an int, a str or a Decimal).
Only counts leave this module: no seed, node id or row.
"""

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from lgk import advisor
from lgk.measure.reference import SCORE_DIGITS

DEFAULT_SAMPLES = 200


def check_samples(samples: Any) -> int:
    """`samples` if it is an int >= the advisor's MIN_SAMPLES (p95 of fewer is the max of a few)."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < advisor.MIN_SAMPLES:
        raise ValueError(f"measure_samples must be an int >= {advisor.MIN_SAMPLES} "
                         f"(the advisor's MIN_SAMPLES), got {samples!r}")
    return samples


def normalize(shape: str, rows: Iterable[Sequence[Any]]) -> List[Tuple[Any, ...]]:
    """Served rows as the reference's sorted tuples.

    web: (hop, node_id, support[, score]) -> (hop, node_id, support)
    tree: (node_id, depth[, score]) -> (depth, node_id)
    similar: (node_id, score) -> (node_id, score rounded)
    """
    if shape == "web":
        return sorted((int(r[0]), str(r[1]), int(r[2])) for r in rows)
    if shape == "tree":
        return sorted((int(r[1]), str(r[0])) for r in rows)
    if shape == "similar":
        return sorted((str(r[0]), round(float(r[1]), SCORE_DIGITS)) for r in rows)
    raise ValueError(f"no normalisation for walk shape {shape!r}")


def check_depth(conn: Any, shape: str, sql: str, seeds: Sequence[str],
                reference: Callable[[str], List[Tuple[Any, ...]]]) -> Dict[str, int]:
    """Run the served `sql` per seed (bound as %(start_node_id)s) and compare to `reference(seed)`."""
    equal = 0
    for seed in seeds:
        served = normalize(shape, conn.execute(sql, {"start_node_id": seed}).fetchall())
        equal += served == sorted(reference(seed))
    return {"seeds_checked": len(seeds), "seeds_equal": equal, "seeds_different": len(seeds) - equal}
