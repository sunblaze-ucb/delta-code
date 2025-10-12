import csv
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import load_sql, load_database_ddl

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|replace|grant|revoke|truncate)\b",
    re.IGNORECASE,
)


# ===================================================================
# Utility Functions
# ===================================================================
def validate_sql_schema(sql: str, database: str) -> None:
    """
    Validate that tables and columns referenced in SQL exist in the database schema.
    Args:
        sql: The SQL statement to validate
        database: Database context as DDL from .sql file
    Raises:
        ValueError: If referenced tables or columns don't exist
    """
    # Extract table names and column references from SQL
    sql_upper = sql.upper()

    # Parse DDL content to extract available tables and their columns
    available_tables = {}
    available_columns = {}

    # Find all CREATE TABLE statements in DDL
    create_table_pattern = r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"\'`]?(\w+)[\"\'`]?\s*\(\s*(.*?)\s*\);"
    table_matches = re.findall(
        create_table_pattern, database, re.IGNORECASE | re.DOTALL
    )

    for table_name, columns_def in table_matches:
        table_name_upper = table_name.upper()
        available_tables[table_name_upper] = table_name

        # Parse column definitions
        columns = []
        # Split by comma, but be careful of commas within constraints
        column_lines = []
        current_line = ""
        paren_depth = 0

        for char in columns_def:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "," and paren_depth == 0:
                column_lines.append(current_line.strip())
                current_line = ""
                continue
            current_line += char

        if current_line.strip():
            column_lines.append(current_line.strip())

        for line in column_lines:
            line = line.strip()
            if not line:
                continue

            # Skip constraint lines (PRIMARY KEY, FOREIGN KEY, etc.)
            if re.match(
                r"^\s*(PRIMARY\s+KEY|FOREIGN\s+KEY|CONSTRAINT|UNIQUE|CHECK)",
                line,
                re.IGNORECASE,
            ):
                continue

            # Extract column name (first quoted or unquoted word)
            col_match = re.match(r"[\"\'`]?(\w+)[\"\'`]?", line)
            if col_match:
                col_name = col_match.group(1).upper()
                columns.append(col_name)

        available_columns[table_name_upper] = columns

    # Extract table names from FROM and JOIN clauses
    referenced_tables = set()
    table_aliases = {}  # alias -> real_table_name

    # FROM clause - handle quoted table names
    from_match = re.search(
        r"FROM\s+[\"'`]?(\w+)[\"'`]?(?:\s+(?:AS\s+)?[\"'`]?(\w+)[\"'`]?)?", sql_upper
    )
    if from_match:
        table_name = from_match.group(1)
        alias = from_match.group(2) if from_match.group(2) else table_name
        referenced_tables.add(table_name)
        table_aliases[alias] = table_name

    # JOIN clauses - handle quoted table names
    join_matches = re.findall(
        r"JOIN\s+[\"'`]?(\w+)[\"'`]?(?:\s+(?:AS\s+)?[\"'`]?(\w+)[\"'`]?)?", sql_upper
    )
    for match in join_matches:
        table_name = match[0]
        alias = match[1] if match[1] else table_name
        referenced_tables.add(table_name)
        table_aliases[alias] = table_name

    # Check if all referenced tables exist
    missing_tables = []
    for table in referenced_tables:
        if table not in available_tables:
            missing_tables.append(table)

    if missing_tables:
        available_table_names = list(available_tables.keys())
        raise ValueError(
            f"SQL references non-existent table(s): {missing_tables}. "
            f"Available tables: {available_table_names}"
        )

    # Extract column references (basic pattern matching)
    # Look for patterns like: table.column, alias.column, or standalone column names
    column_references = set()

    # Pattern for qualified column references (table.column or alias.column) - handle quoted names
    qualified_refs = re.findall(r"[\"'`]?(\w+)[\"'`]?\.[\"'`]?(\w+)[\"'`]?", sql_upper)
    for table_or_alias, column in qualified_refs:
        # Resolve alias to actual table name
        actual_table = table_aliases.get(table_or_alias, table_or_alias)
        column_references.add((actual_table, column))

    # Check if referenced columns exist in their tables
    missing_columns = []
    for table, column in column_references:
        if table in available_columns:
            table_columns = available_columns[table]
            if column not in table_columns and column not in [
                "*",
                "ROWID",
            ]:  # * and ROWID are always valid
                missing_columns.append(f"{table}.{column}")
        # If table doesn't exist, it would have been caught in the table check above

    if missing_columns:
        # Provide helpful information about available columns
        column_info = []
        for table in referenced_tables:
            if table in available_columns:
                cols = available_columns[table][:5]  # Show first 5 columns
                more = "..." if len(available_columns[table]) > 5 else ""
                column_info.append(f"{table}: {cols}{more}")

        raise ValueError(
            f"SQL references non-existent column(s): {missing_columns}. "
            f"Available columns by table: {column_info}"
        )


def _select_only_guard(sql: str) -> str:
    s = sql.strip()
    # normalize: remove single trailing semicolon
    if s.endswith(";"):
        s = s[:-1].strip()
    # must start with SELECT or WITH
    if not (s.lower().startswith("select") or s.lower().startswith("with")):
        raise ValueError("SQL must be SELECT-only (or WITH … SELECT).")
    # forbid dangerous keywords
    if _FORBIDDEN.search(s):
        raise ValueError("SQL contains forbidden keywords (DDL/DML/PRAGMA/etc.).")
    # single-statement guard
    if ";" in s:
        raise ValueError("SQL must contain a single statement.")
    return s


def _normalize_colnames(raw_cols: List[Optional[str]]) -> List[str]:
    """Ensure non-empty, unique column names; fallback to col1, col2, ..."""
    cols: List[str] = []
    seen = set()
    for i, name in enumerate(raw_cols, start=1):
        base = (name or "").strip() or f"col{i}"
        candidate = base
        k = 2
        while candidate in seen:
            candidate = f"{base}_{k}"
            k += 1
        cols.append(candidate)
        seen.add(candidate)
    return cols


def _infer_sqlite_type(value: Any) -> str:
    """
    Infer a coarse SQLite type label from a Python value for signature purposes.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "INTEGER"  # SQLite stores booleans as integers
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "BLOB"
    # Everything else → TEXT
    return "TEXT"


def _get_database_from_query_bank(query_bank_record: Dict[str, Any]) -> str:
    """
    Extract database basename from query_bank record.
    """
    db_name = query_bank_record.get("database_used")
    if isinstance(db_name, str) and db_name.endswith(".sqlite"):
        db_name = db_name[:-7] + ".sql"
    if not isinstance(db_name, str) or not db_name:
        raise ValueError(f"Invalid 'database_used' in query_bank: {db_name}")
    return str(db_name)


def canonical_cell(value: Any) -> str:
    """
    Deterministic string for hashing & sorting.
    - NULL -> "NULL"
    - int/bool -> decimal string
    - float -> 15 significant digits (locale-agnostic)
    - bytes -> hex with 0x prefix
    - str -> as-is
    - other -> str(value)
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.15g}"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "0x" + bytes(value).hex()
    if isinstance(value, str):
        return value
    return str(value)


def sort_key(row: List[Any]) -> tuple:
    """
    Produce a stable sort key tuple for a row when order is NOT sensitive.
    Order across types: NULL < numbers < text < blob < other
    """

    def key_for(v: Any) -> tuple:
        if v is None:
            return (0, "")
        if isinstance(v, (bool, int, float)):
            # use canonical string to avoid NaN/inf surprises
            return (1, canonical_cell(v))
        if isinstance(v, str):
            return (2, v)
        if isinstance(v, (bytes, bytearray, memoryview)):
            return (3, "0x" + bytes(v).hex())
        return (4, canonical_cell(v))

    return tuple(key_for(v) for v in row)


def normalize_rows(
    *,
    rows: List[tuple],
    order_sensitive: bool,
) -> List[List[Any]]:
    """
    Return a normalized list of row lists; stable order if order_sensitive=False.
    """
    rows_list = [list(r) for r in rows]
    if not order_sensitive:
        rows_list.sort(key=sort_key)
    return rows_list


# ===================================================================
# Main Pipeline
# ===================================================================
def verify_format(
    query_id: str,
    database_path: Union[str, Path],
    query_bank_file: Union[str, Path],
    groundtruth_bank_file: Union[str, Path],
    result_db_path: Union[str, Path],
) -> None:
    """
    Execute SQL, materialize results, generate signature.
    Main pipeline:
      - Load SQL from groundtruth bank file and actual database by name
      - Validate SQL schema and produce a <id>.sql file for execution
      - Execute SQL against source database (read-only)
      - Materialize signature and FULL result into ./result/verdict/<id>.sqlite
    """
    # 1. Load SQL and database
    qbank = Path(query_bank_file)
    if not qbank.exists():
        raise FileNotFoundError(f"query_bank not found: {qbank}")

    query_record = None
    with qbank.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if obj.get("id") == query_id:
                        query_record = obj
                        break
                except Exception:
                    continue

    if query_record is None:
        raise KeyError(f"id '{query_id}' not found in {qbank}")

    # Get database name and resolve paths
    database_name = _get_database_from_query_bank(query_record)
    if not database_name.endswith(".sql"):
        raise ValueError(f"Expected .sql file in database_used, got: {database_name}")

    sqlite_name = database_name.replace(".sql", ".sqlite")
    db_root = Path(database_path)

    # Find DDL file for validation
    ddl_matches = sorted(p for p in db_root.rglob(database_name) if p.is_file())
    if not ddl_matches:
        raise FileNotFoundError(
            f"DDL file not found: '{database_name}' under {db_root}"
        )

    # Find SQLite file for execution
    sqlite_matches = sorted(p for p in db_root.rglob(sqlite_name) if p.is_file())
    if not sqlite_matches:
        raise FileNotFoundError(f"Database not found: '{sqlite_name}' under {db_root}")

    # 2. validate SQL
    sql_raw = load_sql(query_id=query_id, groundtruth_bank_file=groundtruth_bank_file)
    sql_clean = _select_only_guard(sql_raw)
    database_ddl = load_database_ddl(str(ddl_matches[0]))
    validate_sql_schema(sql_clean, database_ddl)

    # Save validated SQL to verdict directory
    result_dir = Path(result_db_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    sql_file = result_dir / f"{query_id}.sql"
    sql_file.write_text(sql_clean + "\n", encoding="utf-8")

    result_db_file = result_dir / f"{query_id}.sqlite"
    if result_db_file.exists():
        result_db_file.unlink()

    src_uri = f"{sqlite_matches[0].resolve().as_uri()}?mode=ro"

    try:
        # 3. execute SQL
        with sqlite3.connect(src_uri, uri=True, timeout=10.0) as src_conn:
            src_cur = src_conn.cursor()
            src_cur.execute(sql_clean)

            # Get column info
            raw_cols = [
                d[0] if d and len(d) > 0 else None for d in (src_cur.description or [])
            ]
            columns = _normalize_colnames(raw_cols)

            # 4. materialize results
            with sqlite3.connect(str(result_db_file)) as dst_conn:
                dst_cur = dst_conn.cursor()

                # Create result table
                col_defs = ", ".join([f'"{c}"' for c in columns])
                dst_cur.execute(f'CREATE TABLE "result" ({col_defs})')

                # Create meta table
                dst_cur.execute(
                    'CREATE TABLE "meta" ('
                    "id TEXT, source_db TEXT, executed_sql TEXT, "
                    "checksum TEXT, col_names_json TEXT, row_count INTEGER)"
                )

                # Stream all results
                row_count = 0
                placeholders = ", ".join(["?"] * len(columns)) if columns else ""

                while True:
                    batch = src_cur.fetchmany(1000)
                    if not batch:
                        break
                    dst_cur.executemany(
                        f'INSERT INTO "result" VALUES ({placeholders})', batch
                    )
                    row_count += len(batch)

                # Compute signature
                order_sensitive = bool(
                    re.search(r"\bORDER\s+BY\b", sql_clean, re.IGNORECASE)
                )

                if order_sensitive:
                    dst_cur.execute('SELECT * FROM "result" ORDER BY rowid')
                else:
                    dst_cur.execute('SELECT * FROM "result"')

                all_rows = dst_cur.fetchall()
                rows_norm = normalize_rows(
                    rows=all_rows, order_sensitive=order_sensitive
                )

                # Compute checksum
                sio = StringIO()
                writer = csv.writer(sio, lineterminator="\n")
                for row in rows_norm:
                    writer.writerow([canonical_cell(v) for v in row])
                checksum = (
                    "sha256:"
                    + hashlib.sha256(sio.getvalue().encode("utf-8")).hexdigest()
                )

                # Infer column types
                col_types = []
                for idx, name in enumerate(columns):
                    col_type = "NULL"
                    for row in rows_norm:
                        if idx < len(row) and row[idx] is not None:
                            col_type = _infer_sqlite_type(row[idx])
                            break
                    col_types.append({"name": name, "type": col_type})

                # Save metadata
                dst_cur.execute(
                    'INSERT INTO "meta" VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        query_id,
                        database_name,
                        sql_clean,
                        checksum,
                        json.dumps(col_types, ensure_ascii=False),
                        row_count,
                    ),
                )

                dst_conn.commit()

    except sqlite3.Error as e:
        raise RuntimeError(f"SQL execution error: {e}")
