#!/usr/bin/env python3
"""
Smart cleanup script for SQL generation pipeline results.

This script removes/clears only FAILED results (those not present in dataset.jsonl):
- Entries in all .jsonl files for failed IDs
- .sql and .sqlite files in ./result/verdict/ for failed IDs

Successful results (present in dataset.jsonl) are preserved.

Usage:
    python cleanup_pipeline_results.py [--force]
    
Options:
    --force    Skip confirmation prompt and clean immediately
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set


def load_successful_ids() -> Set[str]:
    """Load successful IDs from dataset.jsonl."""
    successful_ids = set()
    dataset_path = Path("./result/dataset.jsonl")
    
    if not dataset_path.exists():
        print("ℹ️  No dataset.jsonl found - will remove all pipeline results")
        return successful_ids
    
    try:
        with dataset_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get('id')
                    if record_id:
                        successful_ids.add(record_id)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Invalid JSON on line {line_num} in dataset.jsonl: {e}")
                    continue
    except Exception as e:
        print(f"❌ Error reading dataset.jsonl: {e}")
        return set()
    
    return successful_ids


def confirm_cleanup(successful_ids: Set[str]) -> bool:
    """Ask user for confirmation before cleaning up files."""
    print("⚠️  This will permanently delete/clear FAILED pipeline results only:")
    print("   • Entries in all .jsonl files (for failed IDs)")
    print("   • .sql and .sqlite files in ./result/verdict/ (for failed IDs)")
    print()
    print(f"✅ Successful results will be preserved ({len(successful_ids)} IDs found in dataset.jsonl)")
    if successful_ids:
        print(f"   Successful IDs: {', '.join(list(successful_ids)[:3])}{'...' if len(successful_ids) > 3 else ''}")
    print()
    
    response = input("Are you sure you want to proceed? (y/N): ").strip().lower()
    return response in ['y', 'yes']


def cleanup_files_by_id(file_paths: List[Path], successful_ids: Set[str], description: str) -> int:
    """Remove files for failed IDs only (those not in successful_ids) and return count."""
    removed_count = 0
    for file_path in file_paths:
        try:
            if file_path.exists():
                # Extract ID from filename (without extension)
                file_id = file_path.stem
                if file_id not in successful_ids:
                    file_path.unlink()
                    removed_count += 1
                    print(f"   ✅ Removed failed result: {file_path}")
                else:
                    print(f"   ⏭️  Preserved successful result: {file_path}")
        except Exception as e:
            print(f"   ❌ Failed to remove {file_path}: {e}")
    
    if removed_count > 0:
        print(f"   🗑️  {description}: {removed_count} failed files removed")
    else:
        print(f"   ℹ️  {description}: No failed files found to remove")
    
    return removed_count


def filter_jsonl_files(jsonl_paths: List[Path], successful_ids: Set[str], description: str) -> int:
    """Filter JSONL files to keep only successful IDs and return count of files processed."""
    processed_count = 0
    for jsonl_path in jsonl_paths:
        try:
            if not jsonl_path.exists():
                continue
                
            # Read existing records
            records = []
            with jsonl_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        record_id = record.get('id')
                        if record_id and record_id in successful_ids:
                            records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  Warning: Invalid JSON on line {line_num} in {jsonl_path}: {e}")
                        continue
            
            # Rewrite file with only successful records
            with jsonl_path.open('w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            processed_count += 1
            print(f"   ✅ Filtered: {jsonl_path} ({len(records)} successful records kept)")
            
        except Exception as e:
            print(f"   ❌ Failed to filter {jsonl_path}: {e}")
    
    if processed_count > 0:
        print(f"   🧹 {description}: {processed_count} files filtered")
    else:
        print(f"   ℹ️  {description}: No files found to filter")
    
    return processed_count


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up failed SQL generation pipeline results (preserve successful ones)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Skip confirmation prompt and clean immediately"
    )
    
    args = parser.parse_args()
    
    # Check if result directory exists
    result_dir = Path("./result")
    if not result_dir.exists():
        print("ℹ️  No ./result directory found. Nothing to clean up.")
        sys.exit(0)
    
    print("🧹 Smart SQL Pipeline Results Cleanup")
    print("=" * 50)
    
    # Load successful IDs from dataset.jsonl
    print("📊 Loading successful results from dataset.jsonl...")
    successful_ids = load_successful_ids()
    print(f"   Found {len(successful_ids)} successful results to preserve")
    
    # Ask for confirmation unless --force is used
    if not args.force:
        if not confirm_cleanup(successful_ids):
            print("❌ Cleanup cancelled.")
            sys.exit(0)
        print()
    
    print("🚀 Starting smart cleanup (preserving successful results)...")
    total_removed = 0
    total_filtered = 0
    
    # 1. Filter all JSONL files to remove failed IDs
    print("\n1️⃣ Filtering all JSONL files...")
    groundtruth_dir = result_dir / "groundtruth"
    query_dir = result_dir / "query"
    verdict_dir = result_dir / "verdict"
    
    all_jsonl_files = []
    
    # Collect all JSONL files from all directories
    if groundtruth_dir.exists():
        all_jsonl_files.extend([
            groundtruth_dir / "groundtruth_backward.jsonl",
            groundtruth_dir / "groundtruth_forward.jsonl"
        ])
    
    if query_dir.exists():
        all_jsonl_files.extend([
            query_dir / "query_backward.jsonl",
            query_dir / "query_forward.jsonl"
        ])
    
    if verdict_dir.exists():
        all_jsonl_files.extend([
            verdict_dir / "verdict_backward.jsonl",
            verdict_dir / "verdict_forward.jsonl"
        ])
    
    total_filtered += filter_jsonl_files(all_jsonl_files, successful_ids, "All JSONL files")
    
    # 2. Remove failed .sql and .sqlite files in ./result/verdict/
    print("\n2️⃣ Cleaning failed SQL and SQLite files in verdict directory...")
    if verdict_dir.exists():
        sql_files = list(verdict_dir.glob("*.sql"))
        sqlite_files = list(verdict_dir.glob("*.sqlite"))
        all_verdict_files = sql_files + sqlite_files
        total_removed += cleanup_files_by_id(all_verdict_files, successful_ids, "SQL and SQLite files in verdict")
    else:
        print("   ℹ️  Verdict directory not found")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Smart cleanup completed!")
    print(f"📊 Summary:")
    print(f"   • SQL and SQLite files removed from verdict: {total_removed}")
    print(f"   • JSONL files filtered: {total_filtered}")
    print(f"   • Successful results preserved: {len(successful_ids)}")
    print(f"   • Total operations: {total_removed + total_filtered}")
    
    if total_removed + total_filtered == 0:
        print("\n🎉 No cleanup needed - no failed results found!")
    else:
        print(f"\n🎉 Successfully cleaned up {total_removed + total_filtered} failed results!")
        print("💡 Successful results have been preserved for future use.")


if __name__ == "__main__":
    main()
