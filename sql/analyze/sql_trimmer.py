#!/usr/bin/env python3
"""
sql_trimmer.py — Clean SQL file trimmer that processes .sql files by:
1. Removing comments (lines starting with --)
2. Randomly sampling rows per table (INSERT INTO statements)
3. Checking character limits and truncating if needed
4. Saving results to output directory

Usage:
  python sql_trimmer.py --input-dir database --output-dir changed --rows-per-table 3
  python sql_trimmer.py --input database/file.sql --output changed/file.sql --rows-per-table 3
"""

import argparse
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any


class SQLTrimmer:
    """Clean SQL file processor that trims files according to specified rules."""
    
    def __init__(self, rows_per_table: int = 3, char_limit: int = 20000, seed: int = 0):
        self.rows_per_table = rows_per_table
        self.char_limit = char_limit
        self.seed = seed
        random.seed(seed)
    
    def remove_comments(self, content: str) -> str:
        """Remove all lines starting with -- (comments)."""
        lines = content.splitlines()
        filtered_lines = [line for line in lines if not line.strip().startswith('--')]
        return '\n'.join(filtered_lines)
    
    def parse_sql_content(self, content: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        Parse SQL content preserving the original order and grouping.
        
        Returns:
            (ordered_statements, table_inserts) where:
            - ordered_statements is a list of dicts with 'type' and 'content'/'table_name'
            - table_inserts[table_name] = [insert_statements] for sampling
        """
        ordered_statements = []
        table_inserts = {}
        current_table = None
        
        # Split content by semicolons to get individual statements
        statements = [stmt.strip() for stmt in content.split(';') if stmt.strip()]
        
        for statement in statements:
            # Check if it's a CREATE TABLE statement
            create_match = re.match(r'CREATE\s+TABLE\s+["\']?(\w+)["\']?', statement, re.IGNORECASE)
            if create_match:
                table_name = create_match.group(1).lower()
                current_table = table_name
                ordered_statements.append({
                    'type': 'create_table',
                    'content': statement + ';',
                    'table_name': table_name
                })
                # Initialize empty insert list for this table
                if table_name not in table_inserts:
                    table_inserts[table_name] = []
                continue
            
            # Check if it's an INSERT INTO statement
            insert_match = re.match(r'INSERT\s+INTO\s+["`\']?(\w+)["`\']?', statement, re.IGNORECASE)
            if insert_match:
                table_name = insert_match.group(1).lower()
                
                # Add placeholder for insert group if this is the first insert for this table
                if table_name not in table_inserts:
                    table_inserts[table_name] = []
                
                # Check if we need to add an insert group placeholder
                table_has_insert_group = any(
                    stmt.get('type') == 'insert_group' and stmt.get('table_name') == table_name 
                    for stmt in ordered_statements
                )
                
                if not table_has_insert_group:
                    ordered_statements.append({
                        'type': 'insert_group',
                        'table_name': table_name
                    })
                
                # Store the insert statement for sampling
                table_inserts[table_name].append(statement + ';')
                continue
            
            # All other statements (non-CREATE TABLE, non-INSERT INTO)
            ordered_statements.append({
                'type': 'other',
                'content': statement + ';'
            })
        
        return ordered_statements, table_inserts
    
    def sample_table_rows(self, table_inserts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Randomly sample rows_per_table INSERT statements from each table."""
        sampled_inserts = {}
        
        for table_name, inserts in table_inserts.items():
            if self.rows_per_table == 0 or len(inserts) == 0:
                sampled_inserts[table_name] = []
            elif len(inserts) <= self.rows_per_table:
                sampled_inserts[table_name] = inserts.copy()
            else:
                sampled_inserts[table_name] = random.sample(inserts, self.rows_per_table)
        
        return sampled_inserts
    
    def build_sql_content(self, ordered_statements: List[Dict[str, Any]], table_inserts: Dict[str, List[str]], 
                         original_counts: Dict[str, int], truncated: bool = False) -> str:
        """Build the final SQL content preserving original order with proper comments."""
        content_parts = []
        
        for statement in ordered_statements:
            if statement['type'] == 'create_table':
                # Add CREATE TABLE statement
                content_parts.append(statement['content'])
                
            elif statement['type'] == 'insert_group':
                # Add INSERT statements for this table with appropriate comments
                table_name = statement['table_name']
                inserts = table_inserts.get(table_name, [])
                original_count = original_counts.get(table_name, 0)
                
                if not truncated and inserts:
                    # Add sampling comment and the sampled inserts
                    content_parts.append(f"-- Randomly sampled {len(inserts)} of {original_count} row(s) from \"{table_name}\"")
                    content_parts.extend(inserts)
                # If truncated or no inserts, add nothing (will add global comment at end if truncated)
                
            elif statement['type'] == 'other':
                # Add other DDL statements (PRAGMA, CREATE INDEX, etc.)
                content_parts.append(statement['content'])
        
        # Add global truncation comment at the end if all tables were truncated
        if truncated:
            content_parts.append("-- All tables' rows truncated for context length purpose. Data still exists in the actual database")
        
        return '\n'.join(content_parts)
    
    def process_file(self, input_path: Path, output_path: Path) -> None:
        """Process a single SQL file according to the trimming rules."""
        print(f"Processing: {input_path.name}")
        
        # Read input file
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {input_path}: {e}")
            return
        
        # Step 1: Remove comments
        content_no_comments = self.remove_comments(content)
        
        # Step 2: Parse SQL content
        ordered_statements, table_inserts = self.parse_sql_content(content_no_comments)
        
        # Store original counts for comments
        original_counts = {table: len(inserts) for table, inserts in table_inserts.items()}
        
        # Count different statement types for reporting
        create_tables = [s for s in ordered_statements if s['type'] == 'create_table']
        other_ddl = [s for s in ordered_statements if s['type'] == 'other']
        
        print(f"  Found {len(create_tables)} CREATE TABLE statements")
        print(f"  Found {len(other_ddl)} other DDL statements")
        print(f"  Found {len(table_inserts)} tables with data:")
        for table, inserts in table_inserts.items():
            print(f"    {table}: {len(inserts)} rows")
        
        # Step 2: Sample rows per table
        sampled_inserts = self.sample_table_rows(table_inserts)
        
        # Step 3: Check character limit
        temp_content = self.build_sql_content(ordered_statements, sampled_inserts, original_counts)
        char_count = len(temp_content)
        
        print(f"  Character count after sampling: {char_count:,}")
        
        # Step 3: Truncate if exceeds limit
        if char_count > self.char_limit:
            print(f"  Exceeds limit ({self.char_limit:,}), truncating all INSERT statements")
            # Remove all INSERT statements, keep only DDL and truncation comments
            truncated_inserts = {table: [] for table in table_inserts.keys()}
            final_content = self.build_sql_content(ordered_statements, truncated_inserts, original_counts, truncated=True)
        else:
            final_content = temp_content
        
        # Step 4: Save to output (replace if exists)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists and notify about replacement
            if output_path.exists():
                print(f"  Replacing existing file: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(final_content)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> None:
        """Process all .sql files in input directory."""
        print(f"Processing directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Settings: {self.rows_per_table} rows per table, {self.char_limit:,} char limit")
        
        # Find all SQL files
        sql_files = list(input_dir.rglob('*.sql'))
        
        if not sql_files:
            print(f"No .sql files found in {input_dir}")
            return
        
        print(f"Found {len(sql_files)} SQL files to process\n")
        
        success_count = 0
        for i, sql_file in enumerate(sql_files, 1):
            print(f"[{i}/{len(sql_files)}] ", end="")
            
            # Calculate output path (preserve relative structure)
            try:
                rel_path = sql_file.relative_to(input_dir)
                output_path = output_dir / rel_path
                
                # Use file-specific seed for reproducible but varied results
                file_seed = self.seed + hash(str(sql_file))
                old_seed = random.getstate()
                random.seed(file_seed)
                
                self.process_file(sql_file, output_path)
                
                # Restore original random state
                random.setstate(old_seed)
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {sql_file}: {e}")
        
        print(f"\nCompleted: {success_count}/{len(sql_files)} files processed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean SQL file trimmer that removes comments, samples rows, and enforces character limits."
    )
    
    # Input/Output
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path, help="Input directory containing .sql files")
    group.add_argument("--input", type=Path, help="Single input .sql file")
    
    parser.add_argument("--output-dir", type=Path, default=Path("changed"), 
                       help="Output directory (default: changed)")
    parser.add_argument("--output", type=Path, help="Output file (for single file mode)")
    
    # Processing options
    parser.add_argument("--rows-per-table", type=int, default=3,
                       help="Number of rows to keep per table (default: 3)")
    parser.add_argument("--char-limit", type=int, default=20000,
                       help="Character limit for files (default: 20000)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducible results (default: 0)")
    
    args = parser.parse_args()
    
    # Validation
    if args.rows_per_table < 0:
        print("Error: rows-per-table must be non-negative")
        return
    
    if args.input and not args.output:
        print("Error: --output required when using --input")
        return
    
    # Create trimmer instance
    trimmer = SQLTrimmer(
        rows_per_table=args.rows_per_table,
        char_limit=args.char_limit,
        seed=args.seed
    )
    
    try:
        if args.input_dir:
            # Directory mode
            if not args.input_dir.exists():
                print(f"Error: Input directory '{args.input_dir}' does not exist")
                return
            trimmer.process_directory(args.input_dir, args.output_dir)
        else:
            # Single file mode
            if not args.input.exists():
                print(f"Error: Input file '{args.input}' does not exist")
                return
            trimmer.process_file(args.input, args.output)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()