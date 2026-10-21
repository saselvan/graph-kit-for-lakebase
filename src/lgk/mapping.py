"""The graph_mapping.yaml contract: what `propose` writes, the interview edits, `build` reads.

```yaml
source_catalog: <UC catalog of the gold tables>
source_schema: <UC schema of the gold tables>
nodes:
  <node_type>:
    source_table: <table>
    natural_key: <column> | [<column>, ...]   # a list = composite key, order matters
    label: <column>
    properties: [<column>, ...]                # optional, default []
    walk_keys: [<property>, ...]               # optional, default []: the properties a
                                               # set walk may test by equality (at most 4)
    joins:                                     # optional: properties from other tables
      - table: <table>                         # a table of source_schema
        key: <column>                          # its column equal to the node's (single) natural_key
        columns: [<column>, ...]               # one-to-one join, OR:
        aggregates: ["sum(<col>) as <name>"]   # sum | min | max | count | count_distinct
        subject: <column>                      # a folded care-subject column (aggregates only)
edges:
  <edge_type>:
    source_table: <table>
    src: <column>
    src_type: <node_type declared under nodes>
    dst: <column>
    dst_type: <node_type declared under nodes>
    fold_subject: <column>   # optional: a folded care-subject column; the edge is
                             # weighted by the DISTINCT subjects behind each (src, dst) pair
```

A fold edge with `src == dst` is the co-membership form (members sharing subjects, both
directions); with `src != dst` it is the pair form (GROUP BY src, dst).

Every name is a plain lower-case SQL identifier (lgk.ident), because build splices names into
Spark SQL and Postgres folds unquoted names to lower case.
Unknown keys are rejected. Every error names the offending field by its dotted path.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from lgk import ident

_TOP_KEYS = ("source_catalog", "source_schema", "nodes", "edges")
_NODE_KEYS = ("source_table", "natural_key", "label", "properties")
_NODE_OPTIONAL_KEYS = ("walk_keys", "joins")
MAX_WALK_KEYS = 4  # per node type: the function form's static dispatch stays reviewable
_JOIN_KEYS = ("table", "key", "columns", "aggregates", "subject")
# The closed list of deterministic aggregates; anything else does not parse.
AGGREGATE_FUNCS = ("sum", "min", "max", "count", "count_distinct")
# Each name is then checked by lgk.ident (the one identifier rule), so `\w+` here only splits.
_AGGREGATE = re.compile(r"^(sum|min|max|count|count_distinct)\((\w+)\) as (\w+)$", re.ASCII)
MAX_JOINS = 8  # per node type (a bound on build time)
MAX_JOIN_OUTPUTS = 16  # per join (a bound on the properties map size)
# Names a join output may never take: the nodes columns.
RESERVED_OUTPUT_NAMES = ("node_id", "node_type", "natural_key", "label", "properties")
# The prefix of the build's helper columns (`_lgk_key`, `_lgk_subjects`), refused in a mapping.
RESERVED_PREFIX = "_lgk_"
_EDGE_KEYS = ("source_table", "src", "src_type", "dst", "dst_type")
_EDGE_OPTIONAL_KEYS = ("fold_subject",)

NaturalKey = Union[str, List[str]]


class MappingError(ValueError):
    """graph_mapping.yaml does not match the contract; the message names the field."""


@dataclass(frozen=True)
class Aggregate:
    """One `<func>(<column>) as <name>` of a join."""
    func: str
    column: str
    name: str

    def to_text(self) -> str:
        return f"{self.func}({self.column}) as {self.name}"


@dataclass(frozen=True)
class JoinSpec:
    """Node properties from another table of the source schema.

    Exactly one of `columns` (one-to-one) and `aggregates` is non-empty. `subject` (a folded
    care-subject column of `table`) floors every output of an aggregate join."""
    table: str
    key: str
    columns: List[str] = field(default_factory=list)
    aggregates: List[Aggregate] = field(default_factory=list)
    subject: Optional[str] = None

    @property
    def outputs(self) -> List[str]:
        """The property names this join adds, in mapping order."""
        return list(self.columns) if self.columns else [a.name for a in self.aggregates]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"table": self.table, "key": self.key}
        if self.columns:
            out["columns"] = list(self.columns)
        else:
            out["aggregates"] = [a.to_text() for a in self.aggregates]
        if self.subject is not None:
            out["subject"] = self.subject
        return out


@dataclass(frozen=True)
class NodeSpec:
    source_table: str
    natural_key: NaturalKey
    label: str
    properties: List[str] = field(default_factory=list)
    walk_keys: List[str] = field(default_factory=list)  # set-walk equality keys
    joins: List[JoinSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """The contract keys; `walk_keys` and `joins` only when non-empty, so older mappings
        round-trip byte for byte."""
        out: Dict[str, Any] = {"source_table": self.source_table, "natural_key": self.natural_key,
                               "label": self.label, "properties": list(self.properties)}
        if self.walk_keys:
            out["walk_keys"] = list(self.walk_keys)
        if self.joins:
            out["joins"] = [j.to_dict() for j in self.joins]
        return out


@dataclass(frozen=True)
class EdgeSpec:
    source_table: str
    src: str
    src_type: str
    dst: str
    dst_type: str
    fold_subject: Optional[str] = None  # set = a folded (weighted) edge type

    @property
    def fold_form(self) -> Optional[str]:
        """"co_membership" (src == dst), "pair" (src != dst), or None for an unfolded edge."""
        if self.fold_subject is None:
            return None
        return "co_membership" if self.src == self.dst else "pair"

    def to_dict(self) -> Dict[str, Any]:
        """The contract keys; `fold_subject` only when set, so unfolded edges round-trip byte for byte."""
        out: Dict[str, Any] = {k: getattr(self, k) for k in _EDGE_KEYS}
        if self.fold_subject is not None:
            out["fold_subject"] = self.fold_subject
        return out


@dataclass(frozen=True)
class GraphMapping:
    source_catalog: str
    source_schema: str
    nodes: Dict[str, NodeSpec]
    edges: Dict[str, EdgeSpec]

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict in the contract shape (what dump_mapping writes)."""
        return {
            "source_catalog": self.source_catalog,
            "source_schema": self.source_schema,
            "nodes": {name: n.to_dict() for name, n in self.nodes.items()},
            "edges": {name: e.to_dict() for name, e in self.edges.items()},
        }


# ----------------------------------------------------------------------------- field checks
def identifier(value: Any, path: str) -> str:
    """`value` if it is a plain lower-case SQL identifier (lgk.ident), else MappingError naming `path`."""
    try:
        name = ident.check(value, f"field '{path}'")
    except ident.IdentifierError as e:
        raise MappingError(str(e)) from None
    if name.startswith(RESERVED_PREFIX):
        raise MappingError(f"field '{path}' is {name!r}: the {RESERVED_PREFIX!r} prefix is reserved "
                           f"for the build's helper columns")
    return name


def _ident_list(value: Any, path: str) -> List[str]:
    if not isinstance(value, list):
        raise MappingError(f"field '{path}' must be a list of column names, got {value!r}")
    return [identifier(v, f"{path}[{i}]") for i, v in enumerate(value)]


def _natural_key(value: Any, path: str) -> NaturalKey:
    if isinstance(value, list):
        if not value:
            raise MappingError(f"field '{path}' must not be an empty list")
        return _ident_list(value, path)
    return identifier(value, path)


def _section(data: Any, path: str, allowed: tuple, required: tuple) -> Dict[str, Any]:
    """Check `data` is a mapping with only `allowed` keys and every `required` key."""
    if not isinstance(data, dict):
        raise MappingError(f"field '{path or '<root>'}' must be a mapping, got {type(data).__name__}")
    prefix = f"{path}." if path else ""
    for key in data:
        if key not in allowed:
            raise MappingError(f"field '{prefix}{key}' is not part of the mapping contract")
    for key in required:
        if key not in data:
            raise MappingError(f"field '{prefix}{key}' is required")
    return data


def _typed_entries(data: Any, path: str) -> Dict[str, Any]:
    """`nodes` / `edges`: a mapping of <type name> -> spec."""
    if not isinstance(data, dict):
        raise MappingError(
            f"field '{path}' must be a mapping of type name -> spec, got {type(data).__name__}"
        )
    return data


# --------------------------------------------------------------------------- section parsers
def _parse_node(data: Any, path: str) -> NodeSpec:
    d = _section(data, path, _NODE_KEYS + _NODE_OPTIONAL_KEYS, _NODE_KEYS[:3])
    spec = NodeSpec(
        source_table=identifier(d["source_table"], f"{path}.source_table"),
        natural_key=_natural_key(d["natural_key"], f"{path}.natural_key"),
        label=identifier(d["label"], f"{path}.label"),
        properties=_ident_list(d.get("properties", []), f"{path}.properties"),
        walk_keys=_walk_keys(d.get("walk_keys", []), f"{path}.walk_keys"),
        joins=_parse_joins(d.get("joins", []), f"{path}.joins"),
    )
    _check_join_outputs(spec, path)
    return spec


# ------------------------------------------------------------------------------- joins
def _parse_joins(value: Any, path: str) -> List[JoinSpec]:
    if not isinstance(value, list):
        raise MappingError(f"field '{path}' must be a list of joins, got {value!r}")
    if len(value) > MAX_JOINS:
        raise MappingError(f"field '{path}' has {len(value)} joins; at most {MAX_JOINS} per node type")
    return [_parse_join(v, f"{path}[{i}]") for i, v in enumerate(value)]


def _parse_join(data: Any, path: str) -> JoinSpec:
    d = _section(data, path, _JOIN_KEYS, ("table", "key"))
    if ("columns" in d) == ("aggregates" in d):
        raise MappingError(f"field '{path}' needs exactly one of 'columns' and 'aggregates'")
    columns = _ident_list(d["columns"], f"{path}.columns") if "columns" in d else []
    aggregates = _aggregates(d["aggregates"], f"{path}.aggregates") if "aggregates" in d else []
    outputs = columns if "columns" in d else aggregates
    where = f"{path}.columns" if "columns" in d else f"{path}.aggregates"
    if not outputs:
        raise MappingError(f"field '{where}' must not be empty")
    if len(outputs) > MAX_JOIN_OUTPUTS:
        raise MappingError(f"field '{where}' has {len(outputs)} outputs; at most {MAX_JOIN_OUTPUTS} per join")
    subject = identifier(d["subject"], f"{path}.subject") if "subject" in d else None
    if subject is not None and columns:
        raise MappingError(f"field '{path}.subject' applies to an 'aggregates' join only: a "
                           f"one-to-one 'columns' join counts no care subjects")
    return JoinSpec(table=identifier(d["table"], f"{path}.table"),
                    key=identifier(d["key"], f"{path}.key"),
                    columns=columns, aggregates=aggregates, subject=subject)


def _aggregates(value: Any, path: str) -> List[Aggregate]:
    if not isinstance(value, list):
        raise MappingError(f"field '{path}' must be a list of '<func>(<column>) as <name>', got {value!r}")
    return [_aggregate(v, f"{path}[{i}]") for i, v in enumerate(value)]


def _aggregate(value: Any, path: str) -> Aggregate:
    """`<func>(<column>) as <name>`, func in the closed deterministic list (no expression)."""
    m = _AGGREGATE.match(value) if isinstance(value, str) else None
    if m is None:
        raise MappingError(
            f"field '{path}' is {value!r}: an aggregate is '<func>(<column>) as <name>' with func "
            f"one of {', '.join(AGGREGATE_FUNCS)} over one column (non-deterministic aggregates "
            f"and expressions are refused)")
    return Aggregate(func=m.group(1), column=identifier(m.group(2), path),
                     name=identifier(m.group(3), path))


def _output_paths(spec: NodeSpec, path: str) -> List[tuple]:
    """(output name, dotted path) for every join output of the node, in mapping order."""
    out = []
    for i, j in enumerate(spec.joins):
        if j.columns:
            out += [(c, f"{path}.joins[{i}].columns[{k}]") for k, c in enumerate(j.columns)]
        else:
            out += [(a.name, f"{path}.joins[{i}].aggregates[{k}]") for k, a in enumerate(j.aggregates)]
    return out


def _check_join_outputs(spec: NodeSpec, path: str) -> None:
    """Joins need a single-column node key; no output collides with a native property, the
    label, the nodes columns or another output."""
    if spec.joins and isinstance(spec.natural_key, list):
        raise MappingError(f"field '{path}.joins' needs a single-column natural_key; node "
                           f"type with a composite natural_key cannot take joins")
    taken = {name: "a nodes column" for name in RESERVED_OUTPUT_NAMES}
    taken.update({p: "a native property" for p in spec.properties})
    taken[spec.label] = "the label"
    for name, where in _output_paths(spec, path):
        if name in taken:
            raise MappingError(f"field '{where}' outputs {name!r}, which is already {taken[name]}")
        taken[name] = f"the output of {where}"


def _walk_keys(value: Any, path: str) -> List[str]:
    """Identifiers, no duplicates, at most MAX_WALK_KEYS. That each is a property of the node
    type is the privacy gate's check (lgk.privacy.gate.check_walk_keys)."""
    keys = _ident_list(value, path)
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise MappingError(f"field '{path}' lists {dupes} more than once")
    if len(keys) > MAX_WALK_KEYS:
        raise MappingError(f"field '{path}' has {len(keys)} keys; at most {MAX_WALK_KEYS} per node type")
    return keys


def _parse_edge(data: Any, path: str, node_types: Dict[str, NodeSpec]) -> EdgeSpec:
    d = _section(data, path, _EDGE_KEYS + _EDGE_OPTIONAL_KEYS, _EDGE_KEYS)
    spec = EdgeSpec(**{k: identifier(d[k], f"{path}.{k}") for k in _EDGE_KEYS},
                    fold_subject=_fold_subject(d, path))
    _check_fold_form(spec, path)
    for side in ("src_type", "dst_type"):
        if getattr(spec, side) not in node_types:
            raise MappingError(
                f"field '{path}.{side}' names node type {getattr(spec, side)!r}, "
                f"which is not declared under nodes ({sorted(node_types)})"
            )
        if isinstance(node_types[getattr(spec, side)].natural_key, list):
            raise MappingError(
                f"field '{path}.{side}' names node type {getattr(spec, side)!r}, which has a "
                f"composite natural_key; an edge's single {side[:3]} column cannot reach it"
            )
    return spec


def _fold_subject(d: Dict[str, Any], path: str) -> Optional[str]:
    """`fold_subject` (an identifier) when given, else None (an unfolded edge)."""
    if "fold_subject" not in d:
        return None
    return identifier(d["fold_subject"], f"{path}.fold_subject")


def _check_fold_form(spec: EdgeSpec, path: str) -> None:
    """A fold subject is counted, never carried: it differs from src and dst; co-membership
    links members of ONE node type, so src_type must equal dst_type."""
    if spec.fold_subject is None:
        return
    if spec.fold_subject in (spec.src, spec.dst):
        raise MappingError(
            f"field '{path}.fold_subject' is {spec.fold_subject!r}, the same column as src or dst; "
            f"the folded care-subject column is counted, never carried as an edge end"
        )
    if spec.src == spec.dst and spec.src_type != spec.dst_type:
        raise MappingError(
            f"field '{path}.dst_type' is {spec.dst_type!r} but src_type is {spec.src_type!r}: a "
            f"co-membership fold (src == dst) links members of one node type"
        )


def parse_mapping(data: Any) -> GraphMapping:
    """Validate a loaded mapping dict against the contract; raise MappingError naming the field."""
    top = _section(data, "", _TOP_KEYS, _TOP_KEYS[:3])
    nodes = {
        identifier(name, f"nodes.{name}"): _parse_node(spec, f"nodes.{name}")
        for name, spec in _typed_entries(top["nodes"], "nodes").items()
    }
    edges = {
        identifier(name, f"edges.{name}"): _parse_edge(spec, f"edges.{name}", nodes)
        for name, spec in _typed_entries(top.get("edges") or {}, "edges").items()
    }
    return GraphMapping(
        source_catalog=identifier(top["source_catalog"], "source_catalog"),
        source_schema=identifier(top["source_schema"], "source_schema"),
        nodes=nodes,
        edges=edges,
    )


def load_mapping(path: Path) -> GraphMapping:
    """Read and validate graph_mapping.yaml."""
    with open(path) as f:
        return parse_mapping(yaml.safe_load(f))


def dump_mapping(mapping: GraphMapping, path: Path) -> None:
    """Write `mapping` as graph_mapping.yaml in the contract shape (key order kept)."""
    with open(path, "w") as f:
        yaml.safe_dump(mapping.to_dict(), f, sort_keys=False, default_flow_style=False)
