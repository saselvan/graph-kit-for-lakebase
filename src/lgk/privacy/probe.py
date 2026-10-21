"""The reader probe: prove the `reader` role can touch no raw table.

Read from the catalog, not by trying a SELECT: a missing USAGE on a version schema makes a
SELECT fail even when a table grant exists, which would hide that grant. So every raw
relation is checked with has_table_privilege / has_any_column_privilege, for the reader and
for PUBLIC, and the role itself must be a non-superuser that is a member of no other role.
Every relation, function and sequence in every version schema (`<kit>_v<N>`) is raw: EXECUTE
on a function (granted to PUBLIC by default) and USAGE/SELECT/UPDATE on a sequence count too.
"""

from typing import Any, Dict, List, Sequence, Tuple

PRIVILEGES = ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")


class ReaderNotIsolatedError(RuntimeError):
    """The reader can reach a raw table, or is a superuser, or is in another role."""


_RAW_SQL = """
SELECT n.nspname, c.relname
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname ~ %s AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY 1, 2
"""

_TABLE_PRIV_SQL = """
SELECT n.nspname, c.relname, p.priv,
       has_table_privilege(%(role)s, c.oid, p.priv),
       has_table_privilege('public', c.oid, p.priv),
       p.priv IN ('SELECT', 'INSERT', 'UPDATE', 'REFERENCES')
         AND has_any_column_privilege(%(role)s, c.oid, p.priv)
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
CROSS JOIN unnest(%(privs)s::text[]) AS p(priv)
WHERE n.nspname ~ %(pattern)s AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY 1, 2, 3
"""

_FUNCTION_PRIV_SQL = """
SELECT n.nspname, p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')',
       has_function_privilege(%(role)s, p.oid, 'EXECUTE'),
       has_function_privilege('public', p.oid, 'EXECUTE')
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname ~ %(pattern)s
ORDER BY 1, 2
"""

_SEQUENCE_PRIV_SQL = """
SELECT n.nspname, c.relname, p.priv,
       has_sequence_privilege(%(role)s, c.oid, p.priv),
       has_sequence_privilege('public', c.oid, p.priv)
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
CROSS JOIN unnest(ARRAY['USAGE', 'SELECT', 'UPDATE']) AS p(priv)
WHERE n.nspname ~ %(pattern)s AND c.relkind = 'S'
ORDER BY 1, 2, 3
"""

_ROLE_SQL = "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s"

_MEMBER_SQL = """
SELECT g.rolname FROM pg_auth_members m
JOIN pg_roles g ON g.oid = m.roleid
JOIN pg_roles r ON r.oid = m.member
WHERE r.rolname = %s ORDER BY 1
"""


def _version_pattern(kit_schema: str) -> str:
    return f"^{kit_schema}_v[0-9]+$"


def raw_relations(conn: Any, kit_schema: str) -> List[Tuple[str, str]]:
    """(schema, relation) for every relation in every version schema of this kit."""
    return [tuple(r) for r in conn.execute(_RAW_SQL, (_version_pattern(kit_schema),)).fetchall()]


def _privilege_violations(conn: Any, kit_schema: str, role: str, pattern: str = "") -> List[str]:
    rows = conn.execute(_TABLE_PRIV_SQL, {
        "role": role, "privs": list(PRIVILEGES), "pattern": pattern or _version_pattern(kit_schema),
    }).fetchall()
    found = []
    for schema, rel, priv, by_role, by_public, by_column in rows:
        if by_public:
            found.append(f"PUBLIC has {priv} on raw {schema}.{rel}")
        elif by_role:
            found.append(f"{role} has {priv} on raw {schema}.{rel}")
        elif by_column:
            found.append(f"{role} has column-level {priv} on raw {schema}.{rel}")
    return found


def _routine_violations(conn: Any, kit_schema: str, role: str) -> List[str]:
    """EXECUTE on a raw function (PUBLIC gets it by default) or any right on a raw sequence."""
    params = {"role": role, "pattern": _version_pattern(kit_schema)}
    found = []
    for schema, fn, by_role, by_public in conn.execute(_FUNCTION_PRIV_SQL, params).fetchall():
        if by_public:
            found.append(f"PUBLIC has EXECUTE on raw function {schema}.{fn}")
        elif by_role:
            found.append(f"{role} has EXECUTE on raw function {schema}.{fn}")
    for schema, seq, priv, by_role, by_public in conn.execute(_SEQUENCE_PRIV_SQL, params).fetchall():
        if by_public:
            found.append(f"PUBLIC has {priv} on raw sequence {schema}.{seq}")
        elif by_role:
            found.append(f"{role} has {priv} on raw sequence {schema}.{seq}")
    return found


def _role_violations(conn: Any, role: str) -> List[str]:
    row = conn.execute(_ROLE_SQL, (role,)).fetchone()
    if row is None:
        return [f"role {role} does not exist"]
    found = []
    if row[0]:
        found.append(f"{role} is a superuser")
    if row[1]:
        found.append(f"{role} bypasses row-level security")
    for (group,) in conn.execute(_MEMBER_SQL, (role,)).fetchall():
        found.append(f"{role} is a member of role {group}")
    return found


def reader_violations(conn: Any, kit_schema: str, role: str = "reader") -> List[str]:
    """Every way `role` could reach raw data; an empty list means the reader is isolated."""
    return (_role_violations(conn, role) + _privilege_violations(conn, kit_schema, role)
            + _routine_violations(conn, kit_schema, role))


def assert_reader_isolated(conn: Any, kit_schema: str, role: str = "reader") -> None:
    """Raise unless the reader is isolated; a probe that found no raw table proves nothing."""
    if not raw_relations(conn, kit_schema):
        raise ReaderNotIsolatedError(f"no raw relations under {_version_pattern(kit_schema)}")
    found = reader_violations(conn, kit_schema, role)
    if found:
        raise ReaderNotIsolatedError("; ".join(found))


# ----------------------------------------------------------------------------
# The function read surface: public_reader and coordinator_reader
# ----------------------------------------------------------------------------
_SURFACE_SQL = """
SELECT n.nspname, n.nspname || '.' || p.proname || '(' || oidvectortypes(p.proargtypes) || ')',
       p.prosecdef, coalesce(array_to_string(p.proconfig, ';'), ''),
       has_function_privilege('public', p.oid, 'EXECUTE'),
       array(SELECT r FROM unnest(%(roles)s::text[]) AS r
             WHERE has_function_privilege(r, p.oid, 'EXECUTE') ORDER BY 1)
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = %(served)s OR n.nspname ~ %(pattern)s
ORDER BY 1, 2
"""


def _function_violations(conn: Any, kit_schema: str, served: str, expected: Dict[str, str],
                         roles: Sequence[str], search_path: str, required: Sequence[str] = ()) -> List[str]:
    """EXECUTE exactly as `expected` ({served signature: role}); none on a staged/raw function."""
    rows = conn.execute(_SURFACE_SQL, {
        "roles": list(roles), "served": served, "pattern": _version_pattern(kit_schema),
    }).fetchall()
    found = []
    for schema, sig, secdef, config, by_public, granted in rows:
        allowed = expected.get(sig) if schema == served else None
        if by_public:
            found.append(f"PUBLIC has EXECUTE on {sig}")
        found += [f"{r} has EXECUTE on {sig}" for r in granted if r != allowed]
        if allowed is not None and (not secdef or f"search_path={search_path}" not in config.split(";")):
            found.append(f"{sig} is not SECURITY DEFINER with search_path={search_path} pinned")
    present = {sig for schema, sig, *_ in rows if schema == served}
    found += [f"served function {sig} is missing" for sig in required if sig not in present]
    return found


def function_role_violations(conn: Any, kit_schema: str, served: str, expected: Dict[str, str],
                             roles: Sequence[str], search_path: str,
                             required: Sequence[str] = ()) -> List[str]:
    """Every way a function role (or PUBLIC, or another role) could read past its functions.

    Each function role must be a plain LOGIN role in no other role, with no table right on any
    relation of the served schema or of any version schema (synced tables included), and
    EXECUTE only on the served functions `expected` gives it.
    """
    surface = f"^{served}$|{_version_pattern(kit_schema)}"
    found = []
    present = [r for r in list(roles) + ["reader"] if conn.execute(_ROLE_SQL, (r,)).fetchone() is not None]
    for role in [r for r in roles if r in present]:  # a role that does not exist reaches nothing
        found += (_role_violations(conn, role) + _privilege_violations(conn, kit_schema, role, surface)
                  + _routine_violations(conn, kit_schema, role))
    return found + _function_violations(conn, kit_schema, served, expected, present, search_path, required)


def assert_function_roles_isolated(conn: Any, kit_schema: str, served: str, expected: Dict[str, str],
                                   roles: Sequence[str], search_path: str,
                                   required: Sequence[str] = ()) -> None:
    """Raise on any violation; `required` served signatures must exist (a surface of none proves nothing)."""
    found = function_role_violations(conn, kit_schema, served, expected, roles, search_path, required)
    if found:
        raise ReaderNotIsolatedError("; ".join(found))
