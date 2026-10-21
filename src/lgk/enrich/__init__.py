"""Enrichment: the fixed menu, graph-shape checks, and the Spark batch methods.

Every method runs as Spark SQL on a UC warehouse through a `sql(statement) -> rows`
callable, and writes one of the fixed enrichment tables in `tables.py`.
"""
