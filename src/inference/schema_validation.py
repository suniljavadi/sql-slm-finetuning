from __future__ import annotations

from collections.abc import Iterable

from sqlglot import exp, parse, parse_one
from sqlglot.errors import SqlglotError


def _schema_statements(schema: str) -> list[exp.Create]:
    try:
        statements = parse(schema, read="sqlite")
    except SqlglotError:
        statements = []

    creates = [
        statement
        for statement in statements
        if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Schema)
    ]
    if creates:
        return creates

    compact_creates: list[exp.Create] = []
    for fragment in schema.split(";"):
        fragment = fragment.strip()
        if not fragment:
            continue
        candidate = fragment if fragment.upper().startswith("CREATE ") else f"CREATE TABLE {fragment}"
        try:
            statement = parse_one(candidate, read="sqlite")
        except SqlglotError as exc:
            raise ValueError("Schema must use CREATE TABLE statements or table(columns) definitions") from exc
        if not isinstance(statement, exp.Create) or not isinstance(statement.this, exp.Schema):
            raise ValueError("Schema must use CREATE TABLE statements or table(columns) definitions")
        compact_creates.append(statement)
    if not compact_creates:
        raise ValueError("Schema does not contain any table definitions")
    return compact_creates


def _schema_catalog(schema: str) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    for statement in _schema_statements(schema):
        table_schema = statement.this
        table_name = table_schema.this.name.casefold()
        columns = {
            (column.this.name if isinstance(column, exp.ColumnDef) else column.name).casefold()
            for column in table_schema.expressions
            if isinstance(column, (exp.ColumnDef, exp.Identifier))
        }
        if not table_name or not columns:
            raise ValueError("Each schema table must define at least one column")
        tables[table_name] = columns
    return tables


def _names(nodes: Iterable[exp.Expression]) -> set[str]:
    return {node.name.casefold() for node in nodes if node.name}


def validate_sql_schema(sql: str, schema: str) -> str:
    if not schema.strip():
        return sql
    catalog = _schema_catalog(schema)
    try:
        statements = [statement for statement in parse(sql, read="sqlite") if statement is not None]
    except SqlglotError as exc:
        raise ValueError("Generated SQL is not valid SQLite syntax") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("Generated SQL must contain one read-only query")

    query = statements[0]
    cte_names = _names(query.find_all(exp.CTE))
    subquery_aliases = {
        subquery.alias.casefold()
        for subquery in query.find_all(exp.Subquery)
        if subquery.alias
    }
    table_aliases: dict[str, str] = {}
    for table in query.find_all(exp.Table):
        table_name = table.name.casefold()
        if table_name in cte_names:
            continue
        if table_name not in catalog:
            raise ValueError(f"Generated SQL references unknown table '{table.name}'.")
        table_aliases[table.alias_or_name.casefold()] = table_name
        table_aliases[table_name] = table_name

    output_aliases = _names(query.find_all(exp.Alias))
    all_columns = set().union(*catalog.values())
    for column in query.find_all(exp.Column):
        column_name = column.name.casefold()
        if not column_name or column_name == "*" or column_name in output_aliases:
            continue
        qualifier = column.table.casefold()
        if qualifier in cte_names or qualifier in subquery_aliases:
            continue
        if qualifier:
            table_name = table_aliases.get(qualifier)
            if table_name is None:
                raise ValueError(f"Generated SQL references unknown table or alias '{column.table}'.")
            if column_name not in catalog[table_name]:
                raise ValueError(f"Generated SQL references unknown column '{column.name}' on table '{table_name}'.")
        elif column_name not in all_columns and not column.this.args.get("quoted", False):
            raise ValueError(f"Generated SQL references unknown column '{column.name}'.")
    return sql