#!/usr/bin/env python3
"""
consolidate_spider_databases.py

Usage:
  python consolidate_spider_databases.py --root ./database

copy .sqlite and .sql files from spider to root directory.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from typing import Dict

EXTS = (".sqlite", ".sql")

def pick_files(db_dir: Path) -> Dict[str, Dict[str, Path]]:
    results: Dict[str, Dict[str, Path]] = {}
    if not db_dir.exists():
        return results

    for sub in db_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        found: Dict[str, Path] = {}
        for ext in EXTS:
            candidates = sorted(p for p in sub.glob(f"*{ext}") if p.is_file())
            if len(candidates) > 1:
                print(f"[warn] {db_dir.name}/{name}: multiple '{ext}' files found, choosing '{candidates[0].name}'.")
            if candidates:
                found[ext.lstrip(".")] = candidates[0]
        if found:
            results[name] = found
    return results

def main(root: Path):
    source_database = root / "spider"
    out_dir = root

    database_map = pick_files(source_database)
    print(f"[info] Found {len(database_map)} databases in spider/database.")

    copied_count = 0
    skipped_count = 0

    for db_name, files in database_map.items():
        # Check if both .sqlite and .sql exist
        has_sqlite = "sqlite" in files
        has_sql = "sql" in files

        if not has_sqlite or not has_sql:
            missing = []
            if not has_sqlite:
                missing.append(".sqlite")
            if not has_sql:
                missing.append(".sql")
            print(f"[missing] {db_name}: missing {', '.join(missing)}")
            skipped_count += 1
            continue

        # Copy both files
        sqlite_src = files["sqlite"]
        sql_src = files["sql"]
        
        sqlite_dst = out_dir / f"{db_name}.sqlite"
        sql_dst = out_dir / f"{db_name}.sql"
        
        shutil.copy2(sqlite_src, sqlite_dst)
        shutil.copy2(sql_src, sql_dst)
        
        print(f"[copied] {db_name}: {sqlite_src.name} -> {sqlite_dst.name}")
        print(f"[copied] {db_name}: {sql_src.name} -> {sql_dst.name}")
        copied_count += 1

    print(f"\n[summary]")
    print(f"Copied databases: {copied_count}")
    print(f"Skipped (missing files): {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate Spider DB files into database root.")
    parser.add_argument("--root", type=Path, default=Path("./database"),
                        help="Path to the database root (default: ./database)")
    args = parser.parse_args()
    main(args.root.resolve())
