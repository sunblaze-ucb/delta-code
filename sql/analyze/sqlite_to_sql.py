"""
sqlite_to_sql.py — Dump an .sqlite database to a portable .sql file
that includes:
  • DDL (CREATE TABLE/VIEW/INDEX/TRIGGER)
  • INSERT statements for all rows (including BLOBs) or a *sample* per table
  • AUTOINCREMENT state via sqlite_sequence (when present)

Row sampling controls (for prompt-sized DDL+content contexts):
  --max-inserts-per-table N   Cap the number of INSERT rows per table (default: 5).
  --small-table-all M         If a table has ≤ M rows, dump *all* rows (default: 20).
  --sample-strategy {head,tail,random}  How to choose rows when sampling (default: head).
  --seed INT                  Seed for 'random' strategy (default: 0).

Usage:
  # Single file (sampled content)
  python sqlite_to_sql.py --db in.db --out out.sql --max-inserts-per-table 5 --small-table-all 20 --sample-strategy head

  # Directory (non-recursive)
  python sqlite_to_sql.py --dir /path/dir --out-dir /path/outdir --max-inserts-per-table 5

  # Directory (recursive)
  python sqlite_to_sql.py --dir /path/dir --out-dir /path/outdir --recursive

Notes:
  - Only uses Python stdlib + sqlite3.
  - Skips internal SQLite objects named like "sqlite_%", except `sqlite_sequence`
    which is dumped to preserve AUTOINCREMENT counters.
"""

from __future__ import annotations

import argparse
import sqlite3
import random
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

def ensure_semicolon(sql: str) -> str:
    sql = sql.strip()
    return sql if sql.endswith(";") else sql + ";"

def qident(name: str) -> str:
    # Quote an identifier with double quotes, escaping any internal quotes.
    return '"' + name.replace('"', '""') + '"'

def sql_literal(value, conn: sqlite3.Connection) -> str:
    """
    Return a SQL literal for any Python value using SQLite's own `quote()`,
    which handles proper escaping for strings and blobs, and leaves numbers bare.
    """
    if value is None:
        return "NULL"
    # bytes -> X'ABCD...'
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "X'" + bytes(value).hex() + "'"
    # Delegate to SQLite to quote strings/numbers/dates reliably
    return conn.execute("SELECT quote(?)", (value,)).fetchone()[0]

def get_user_tables(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    # Return (name, sql) for user-created tables (skip internal sqlite_% tables)
    cur = conn.execute(
        "SELECT name, sql FROM sqlite_schema "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [(r[0], r[1]) for r in cur.fetchall()]

def get_views(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute(
        "SELECT sql FROM sqlite_schema WHERE type='view' ORDER BY name"
    )
    return [r[0] for r in cur.fetchall() if r[0]]

def get_indexes(conn: sqlite3.Connection) -> List[str]:
    # Only include indexes with explicit SQL (skip autoindexes with NULL sql)
    cur = conn.execute(
        "SELECT sql FROM sqlite_schema "
        "WHERE type='index' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [r[0] for r in cur.fetchall() if r[0]]

def get_triggers(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute(
        "SELECT sql FROM sqlite_schema WHERE type='trigger' ORDER BY name"
    )
    return [r[0] for r in cur.fetchall() if r[0]]

def get_foreign_keys(conn: sqlite3.Connection) -> List[Tuple[str, str, str, str, str, str]]:
    """
    Get foreign key information for all tables.
    Returns list of (table, column, ref_table, ref_column, on_update, on_delete)
    """
    foreign_keys = []
    tables = [name for name, _ in get_user_tables(conn)]
    
    for table in tables:
        try:
            # Get foreign key info for this table
            cur = conn.execute(f"PRAGMA foreign_key_list({qident(table)})")
            for row in cur.fetchall():
                # row format: (id, seq, table, from, to, on_update, on_delete, match)
                foreign_keys.append((
                    table,                    # source table
                    row[3],                  # source column (from)
                    row[2],                  # referenced table (table)
                    row[4],                  # referenced column (to)
                    row[5],                  # on_update action
                    row[6]                   # on_delete action
                ))
        except sqlite3.OperationalError:
            # Table might not exist or have foreign keys
            continue
    
    return foreign_keys

def get_table_constraints(conn: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    """
    Get table constraints (PRIMARY KEY, UNIQUE, CHECK, etc.).
    Returns list of (table, type, definition)
    """
    constraints = []
    tables = [name for name, _ in get_user_tables(conn)]
    
    for table in tables:
        try:
            # Get table info to find constraints
            cur = conn.execute(f"PRAGMA table_info({qident(table)})")
            columns = cur.fetchall()
            
            # Check for primary key
            pk_columns = [col[1] for col in columns if col[5] > 0]  # col[5] is pk flag
            if len(pk_columns) > 1:  # Composite primary key
                pk_def = f"PRIMARY KEY ({', '.join(qident(c) for c in pk_columns)})"
                constraints.append((table, "PRIMARY KEY", pk_def))
            elif len(pk_columns) == 1:  # Single column primary key
                constraints.append((table, "PRIMARY KEY", f"PRIMARY KEY ({qident(pk_columns[0])})"))
            
            # Get other constraints from table definition
            cur = conn.execute(f"SELECT sql FROM sqlite_schema WHERE type='table' AND name={qident(table)}")
            table_sql = cur.fetchone()
            if table_sql and table_sql[0]:
                # Parse constraints from CREATE TABLE statement
                sql = table_sql[0].upper()
                if "UNIQUE" in sql:
                    constraints.append((table, "UNIQUE", "UNIQUE constraint found in table definition"))
                if "CHECK" in sql:
                    constraints.append((table, "CHECK", "CHECK constraint found in table definition"))
                    
        except sqlite3.OperationalError:
            continue
    
    return constraints

def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f'PRAGMA table_info({qident(table)})')
    cols = [r[1] for r in cur.fetchall()]  # (cid, name, type, notnull, dflt_value, pk)
    return cols

def primary_key_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cols = []
    for cid, name, ctype, notnull, dflt, pk in conn.execute(f'PRAGMA table_info({qident(table)})'):
        if pk:  # pk position > 0
            cols.append(name)
    return cols

def table_row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {qident(table)}").fetchone()[0]

def iter_rows_sampled(conn: sqlite3.Connection, table: str, limit: int, strategy: str, seed: int):
    """
    Yield sampled rows from `table` as sqlite3.Row. Deterministic for head/tail.
    For random, uses Python sampling over rowid set to honor `seed`.
    """
    if limit <= 0:
        # 0 or negative means no rows requested
        return
    pks = primary_key_columns(conn, table)
    order_clause = ""
    if pks:
        order_clause = " ORDER BY " + ", ".join(qident(c) for c in pks)
    else:
        # Fall back to rowid which is stable for ordinary tables
        order_clause = " ORDER BY rowid"

    if strategy == "head":
        sql = f"SELECT * FROM {qident(table)}{order_clause} LIMIT {int(limit)}"
        for row in conn.execute(sql):
            yield row
        return
    if strategy == "tail":
        # Efficient tail via window is not trivial in SQLite; approximate by OFFSET if cheap.
        # We'll compute count and then use LIMIT/OFFSET.
        count = table_row_count(conn, table)
        offset = max(0, count - limit)
        sql = f"SELECT * FROM {qident(table)}{order_clause} LIMIT {int(limit)} OFFSET {int(offset)}"
        for row in conn.execute(sql):
            yield row
        return
    if strategy == "random":
        # Build a list of candidate rowids deterministically seeded,
        # then fetch those rows in PK/rowid order for readability.
        # For very large tables this collects rowids; acceptable for prompt sampling.
        random.seed(seed + hash(table) % (2**32))
        ids = []
        
        # Check if rowid exists by trying to query it
        try:
            # Test if rowid exists
            conn.execute(f"SELECT rowid FROM {qident(table)} LIMIT 1").fetchone()
            # If we get here, rowid exists
            for (rid,) in conn.execute(f"SELECT rowid FROM {qident(table)}"):
                ids.append(rid)
        except sqlite3.OperationalError:
            # rowid doesn't exist, fall back to using primary key or all columns
            if pks:
                # Use primary key columns for ordering
                pk_placeholders = ", ".join(qident(c) for c in pks)
                for row in conn.execute(f"SELECT {pk_placeholders} FROM {qident(table)}"):
                    ids.append(tuple(row))
            else:
                # No rowid and no PK, just use all columns as identifier
                cols = get_columns(conn, table)
                col_placeholders = ", ".join(qident(c) for c in cols)
                for row in conn.execute(f"SELECT {col_placeholders} FROM {qident(table)}"):
                    ids.append(tuple(row))
        if not ids:
            return
        if len(ids) <= limit:
            chosen = ids
        else:
            chosen = random.sample(ids, limit)
        
        # Fetch the actual rows based on the chosen identifiers
        if not chosen:
            return
            
        # Check if we're using rowid or other identifiers
        try:
            # Test if rowid exists
            conn.execute(f"SELECT rowid FROM {qident(table)} LIMIT 1").fetchone()
            # Use rowid-based selection
            placeholders = ",".join(["?"] * len(chosen))
            sql = f"SELECT * FROM {qident(table)} WHERE rowid IN ({placeholders}){order_clause}"
            for row in conn.execute(sql, chosen):
                yield row
        except sqlite3.OperationalError:
            # rowid doesn't exist, use primary key or all columns for selection
            if pks:
                # Use primary key for selection
                pk_conditions = []
                for pk_tuple in chosen:
                    pk_cond = " AND ".join(f"{qident(pk_col)} = ?" for pk_col in pks)
                    pk_conditions.append(f"({pk_cond})")
                where_clause = " OR ".join(pk_conditions)
                sql = f"SELECT * FROM {qident(table)} WHERE {where_clause}{order_clause}"
                # Flatten the chosen tuples for parameter binding
                flat_params = [param for pk_tuple in chosen for param in pk_tuple]
                for row in conn.execute(sql, flat_params):
                    yield row
            else:
                # Use all columns for selection (less efficient but works)
                cols = get_columns(conn, table)
                col_conditions = []
                for col_tuple in chosen:
                    col_cond = " AND ".join(f"{qident(col)} = ?" for col in cols)
                    col_conditions.append(f"({col_cond})")
                where_clause = " OR ".join(col_conditions)
                sql = f"SELECT * FROM {qident(table)} WHERE {where_clause}{order_clause}"
                # Flatten the chosen tuples for parameter binding
                flat_params = [param for col_tuple in chosen for param in col_tuple]
                for row in conn.execute(sql, flat_params):
                    yield row
        return
    raise ValueError(f"Unknown strategy: {strategy}")

def dump_single_db(
    db_path: Path,
    out_path: Path,
    *,
    max_inserts_per_table: int = 5,
    small_table_all: int = 20,
    sample_strategy: str = "head",
    seed: int = 0,
    overwrite: bool = True,
    include_fk: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle existing files based on overwrite setting
    if out_path.exists():
        if not overwrite:
            print(f"[SKIP] {out_path} already exists (use --overwrite to overwrite)")
            return
        else:
            out_path.unlink()  # Remove existing file to ensure clean overwrite
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.text_factory = lambda b: b.decode(errors="replace") if isinstance(b, (bytes, bytearray)) else b

        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            # Add PRAGMA foreign_keys=ON at the top if foreign keys are included
            if include_fk:
                f.write("PRAGMA foreign_keys=ON;\n\n")
            
            # Emit DDL for tables first
            user_tables = get_user_tables(conn)
            for name, sql in user_tables:
                if not sql:
                    # Defensive: if somehow missing, reconstruct minimal CREATE
                    cols = get_columns(conn, name)
                    col_list = ", ".join(qident(c) for c in cols) if cols else "id INTEGER"
                    sql = f"CREATE TABLE {qident(name)} ({col_list})"
                f.write(ensure_semicolon(sql) + "\n")

            f.write("\n-- Data inserts (sampled per table)\n")
            for name, _ in user_tables:
                cols = get_columns(conn, name)
                if not cols:
                    continue
                col_list_sql = ", ".join(qident(c) for c in cols)
                total_rows = conn.execute(f"SELECT COUNT(*) FROM {qident(name)}").fetchone()[0]

                if total_rows == 0:
                    f.write(f"-- (no rows) {qident(name)}\n\n")
                    continue

                # Decide how many to dump
                if total_rows <= small_table_all:
                    limit = total_rows
                    sampling_note = f"-- Dumping all {total_rows} row(s) from {qident(name)} (≤ small_table_all={small_table_all})\n"
                else:
                    limit = max_inserts_per_table if max_inserts_per_table >= 0 else total_rows
                    sampling_note = (f"-- Sampling {limit} of {total_rows} row(s) from {qident(name)} "
                                     f"(strategy={sample_strategy})\n")
                f.write(sampling_note)

                written = 0
                if limit > 0:
                    for row in iter_rows_sampled(conn, name, limit=limit, strategy=sample_strategy, seed=seed):
                        values = ", ".join(sql_literal(row[c], conn) for c in cols)
                        f.write(f"INSERT INTO {qident(name)} ({col_list_sql}) VALUES ({values});\n")
                        written += 1
                if written == 0:
                    f.write(f"-- (rows not emitted due to limit=0) {qident(name)}\n")
                f.write("\n")

            # Preserve AUTOINCREMENT counters if present
            has_seq = conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='sqlite_sequence'"
            ).fetchone()
            if has_seq:
                f.write("-- Preserve AUTOINCREMENT state\n")
                f.write("DELETE FROM sqlite_sequence;\n")
                for tname, seq in conn.execute("SELECT name, seq FROM sqlite_sequence"):
                    lit_name = conn.execute("SELECT quote(?)", (tname,)).fetchone()[0]
                    lit_seq = "NULL" if seq is None else str(int(seq))
                    f.write(f"INSERT INTO sqlite_sequence(name, seq) VALUES ({lit_name}, {lit_seq});\n")
                f.write("\n")

            # Create views after tables (no data dependency)
            views = get_views(conn)
            if views:
                f.write("-- Views\n")
                for vsql in views:
                    f.write(ensure_semicolon(vsql) + "\n")
                f.write("\n")

            # Create indexes after data for speed
            indexes = get_indexes(conn)
            if indexes:
                f.write("-- Indexes\n")
                for isql in indexes:
                    f.write(ensure_semicolon(isql) + "\n")
                f.write("\n")

            # Add foreign key constraints
            if include_fk:
                foreign_keys = get_foreign_keys(conn)
                if foreign_keys:
                    f.write("-- Foreign Key Constraints\n")
                    for table, column, ref_table, ref_column, on_update, on_delete in foreign_keys:
                        # Skip if ref_column is None (some foreign keys don't specify target column)
                        if ref_column is None:
                            f.write(f"-- Skipping FK {table}.{column} -> {ref_table} (no target column specified)\n")
                            continue
                            
                        fk_sql = (f"ALTER TABLE {qident(table)} ADD CONSTRAINT "
                                 f"fk_{table}_{column} FOREIGN KEY ({qident(column)}) "
                                 f"REFERENCES {qident(ref_table)} ({qident(ref_column)})")
                        if on_update and on_update != "NO ACTION":
                            fk_sql += f" ON UPDATE {on_update}"
                        if on_delete and on_delete != "NO ACTION":
                            fk_sql += f" ON DELETE {on_delete}"
                        f.write(ensure_semicolon(fk_sql) + "\n")
                    f.write("\n")

            # Add table constraints information
            constraints = get_table_constraints(conn)
            if constraints:
                f.write("-- Table Constraints\n")
                for table, constraint_type, definition in constraints:
                    f.write(f"-- {table}: {constraint_type} - {definition}\n")
                f.write("\n")

            # Create triggers last (avoid firing during inserts)
            triggers = get_triggers(conn)
            if triggers:
                f.write("-- Triggers\n")
                for tsql in triggers:
                    f.write(ensure_semicolon(tsql) + "\n")
                f.write("\n")

            f.write("COMMIT;\n")
            f.write("PRAGMA foreign_keys=ON;\n")
    finally:
        conn.close()

def dump_dir(
    dir_path: Path,
    out_dir: Path,
    *,
    max_inserts_per_table: int = 5,
    small_table_all: int = 20,
    sample_strategy: str = "head",
    seed: int = 0,
    overwrite: bool = True,
    include_fk: bool = True,
) -> None:
    exts = {".sqlite", ".db", ".db3", ".sqlite3"}
    files = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)

    if not files:
        print(f"[INFO] No SQLite files found in {dir_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for db in files:
        out_path = out_dir / (db.stem + ".sql")
        print(f"[INFO] Dumping {db} -> {out_path}")
        dump_single_db(
            db, out_path,
            max_inserts_per_table=max_inserts_per_table,
            small_table_all=small_table_all,
            sample_strategy=sample_strategy,
            seed=seed,
            overwrite=overwrite,
            include_fk=include_fk,
        )

def main():
    ap = argparse.ArgumentParser(description="Dump SQLite .sqlite/.db to portable .sql (DDL + INSERTs), with optional per-table row sampling for prompt-sized contexts.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--db", type=Path, help="Path to a single SQLite file")
    g.add_argument("--dir", type=Path, help="Path to a directory containing SQLite files")
    ap.add_argument("--out", type=Path, help="Output .sql path (for --db)")
    ap.add_argument("--out-dir", type=Path, help="Output directory (for --dir)")

    ap.add_argument("--max-inserts-per-table", type=int, default=3, help="Cap the number of INSERTs per table when table has more than --small-table-all rows (default: 3)")
    ap.add_argument("--small-table-all", type=int, default=3, help="Dump ALL rows if table has ≤ this many rows (default: 3)")
    ap.add_argument("--sample-strategy", choices=["head","tail","random"], default="random", help="Row sampling strategy when limiting inserts (default: head)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for 'random' sampling strategy (default: 0)")
    ap.add_argument("--no-overwrite", action="store_false", dest="overwrite", help="Skip existing output files instead of overwriting (default: overwrite)")
    ap.add_argument("--no-fk", action="store_false", dest="include_fk", help="Skip foreign key constraints in output (default: include)")

    args = ap.parse_args()

    if args.db:
        if not args.out:
            raise SystemExit("--out is required when using --db")
        dump_single_db(
            args.db, args.out,
            max_inserts_per_table=args.max_inserts_per_table,
            small_table_all=args.small_table_all,
            sample_strategy=args.sample_strategy,
            seed=args.seed,
            overwrite=args.overwrite,
            include_fk=args.include_fk,
        )
        print(f"[DONE] Wrote {args.out}")
    else:
        if not args.out_dir:
            raise SystemExit("--out-dir is required when using --dir")
        dump_dir(
            args.dir, args.out_dir,
            max_inserts_per_table=args.max_inserts_per_table,
            small_table_all=args.small_table_all,
            sample_strategy=args.sample_strategy,
            seed=args.seed,
            overwrite=args.overwrite,
            include_fk=args.include_fk,
        )
        print(f"[DONE] Wrote dumps to {args.out_dir}")

if __name__ == "__main__":
    main()