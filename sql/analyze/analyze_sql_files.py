#!/usr/bin/env python3
"""
Script to analyze .sql files in a database directory and print character counts in descending order
"""

import os
import sys
import argparse
from pathlib import Path

def get_file_character_count(file_path):
    """Get character count of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return len(content)
    except (OSError, IOError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return 0

def find_sql_files(database_dir):
    """Find all .sql files in the database directory"""
    sql_files = []
    
    if not os.path.exists(database_dir):
        print(f"Error: Database directory '{database_dir}' does not exist")
        return sql_files
    
    for root, dirs, files in os.walk(database_dir):
        for file in files:
            if file.lower().endswith('.sql'):
                file_path = os.path.join(root, file)
                # Get relative path from database directory
                rel_path = os.path.relpath(file_path, start=database_dir)
                sql_files.append((file_path, rel_path))
    
    return sql_files

def analyze_sql_files(database_dir, min_chars=0, show_stats=True, top_n=None):
    """Main function to analyze .sql files and print character counts"""
    
    print(f"Analyzing .sql files in '{database_dir}' directory...")
    
    # Find all .sql files
    sql_files = find_sql_files(database_dir)
    
    if not sql_files:
        print(f"No .sql files found in '{database_dir}' directory")
        return
    
    print(f"Found {len(sql_files)} .sql files")
    
    # Analyze each file
    file_stats = []
    total_chars = 0
    
    print("\nAnalyzing files...")
    for i, (file_path, rel_path) in enumerate(sql_files, 1):
        char_count = get_file_character_count(file_path)
        if char_count >= min_chars:
            file_stats.append((rel_path, char_count))
            total_chars += char_count
        
        # Progress indicator
        if i % 50 == 0 or i == len(sql_files):
            print(f"  Processed {i}/{len(sql_files)} files...")
    
    if not file_stats:
        print(f"No files found with at least {min_chars} characters")
        return
    
    # Sort by character count (descending)
    file_stats.sort(key=lambda x: x[1], reverse=True)
    
    # Apply top N filter if specified
    if top_n and top_n > 0:
        file_stats = file_stats[:top_n]
    
    # Print results
    print(f"\n{'='*80}")
    print(f"SQL FILE CHARACTER COUNT ANALYSIS")
    if top_n and top_n > 0:
        print(f"Showing top {top_n} files")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'File Name':<50} {'Characters':<12} {'Size (KB)':<10}")
    print(f"{'-'*80}")
    
    for rank, (rel_path, char_count) in enumerate(file_stats, 1):
        size_kb = char_count / 1024
        # Truncate long filenames for display
        display_name = rel_path if len(rel_path) <= 47 else "..." + rel_path[-44:]
        print(f"{rank:<6} {display_name:<50} {char_count:<12,} {size_kb:<10.1f}")
    
    if show_stats:
        print(f"{'-'*80}")
        print(f"Total files analyzed: {len(file_stats)}")
        print(f"Total characters: {total_chars:,}")
        print(f"Average characters per file: {total_chars // len(file_stats):,}")
        print(f"Largest file: {file_stats[0][0]} ({file_stats[0][1]:,} characters)")
        print(f"Smallest file: {file_stats[-1][0]} ({file_stats[-1][1]:,} characters)")

def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Analyze .sql files and print character counts in descending order')
    parser.add_argument('database_dir', help='Path to database directory')
    parser.add_argument('--min-chars', type=int, default=0, help='Minimum character count to include (default: 0)')
    parser.add_argument('--no-stats', action='store_true', help='Hide summary statistics')
    parser.add_argument('--top', type=int, help='Show only top N files')
    
    args = parser.parse_args()
    
    # Validate database directory
    if not os.path.exists(args.database_dir):
        print(f"Error: Directory '{args.database_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(args.database_dir):
        print(f"Error: '{args.database_dir}' is not a directory")
        sys.exit(1)
    
    # Analyze files
    analyze_sql_files(
        database_dir=args.database_dir,
        min_chars=args.min_chars,
        show_stats=not args.no_stats,
        top_n=args.top
    )

if __name__ == "__main__":
    main()
