"""SQL the generated app reads besides the walks (those come from lgk.steps.walks)."""


def attribution_query(served_schema: str) -> str:
    """The source's licence notice (lgk.attribution), for the app to show with the data.

    `served_schema` is `lgk.versions.served_schema(config)`; there is no default here.
    """
    from lgk import ident
    from lgk.attribution import COLUMNS, VIEW

    return f"SELECT {', '.join(COLUMNS)} FROM {ident.check(served_schema, 'served_schema')}.{VIEW}"
