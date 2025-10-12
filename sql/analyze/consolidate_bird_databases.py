#!/usr/bin/env python3
"""
consolidate_bird_databases.py

Usage:
  python consolidate_bird_databases.py --root ./bird --dest ./compile2

Copy .sqlite files from subdirectories to root directory.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from typing import Dict

def pick_sqlite_files(db_dir: Path) -> Dict[str, Path]:
    results: Dict[str, Path] = {}
    if not db_dir.exists():
        return results

    for sub in db_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name
        
        # Look for .sqlite files in this subdirectory
        sqlite_files = sorted(p for p in sub.glob("*.sqlite") if p.is_file())
        if len(sqlite_files) > 1:
            print(f"[warn] {db_dir.name}/{name}: multiple .sqlite files found, choosing '{sqlite_files[0].name}'.")
        if sqlite_files:
            results[name] = sqlite_files[0]
    
    return results

def main(source: Path, dest: Path):
    database_map = pick_sqlite_files(source)
    print(f"[info] Found {len(database_map)} subdirectories with .sqlite files.")

    copied_count = 0
    skipped_count = 0

    # Get all subdirectories to check for missing .sqlite files
    all_subdirs = [sub.name for sub in source.iterdir() if sub.is_dir()]
    missing_sqlite = []

    for subdir_name in all_subdirs:
        if subdir_name in database_map:
            # Copy the .sqlite file
            sqlite_src = database_map[subdir_name]
            sqlite_dst = dest / f"{subdir_name}.sqlite"
            
            shutil.copy2(sqlite_src, sqlite_dst)
            print(f"[copied] {subdir_name}: {sqlite_src.name} -> {sqlite_dst.name}")
            copied_count += 1
        else:
            # Report missing .sqlite file
            missing_sqlite.append(subdir_name)
            print(f"[missing] {subdir_name}: no .sqlite file found")
            skipped_count += 1

    print(f"\n[summary]")
    print(f"Copied databases: {copied_count}")
    print(f"Skipped (missing .sqlite): {skipped_count}")
    if missing_sqlite:
        print(f"Missing .sqlite files in: {', '.join(missing_sqlite)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate Bird DB files into destination directory.")
    parser.add_argument("--source", type=Path, default=Path("./bird"),
                        help="Path to the source directory (default: ./bird)")
    parser.add_argument("--dest", type=Path, default=Path("./compile2"),
                        help="Path to the destination directory (default: ./compile2)")
    args = parser.parse_args()
    main(args.source.resolve(), args.dest.resolve())
