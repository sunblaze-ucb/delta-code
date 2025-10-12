import json
import pathlib
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import (
    persist_result,
    load_database_ddl,
    load_query,
    load_groundtruth_record,
    load_result_db,
)


# ===================================================================
# Utility Functions
# ===================================================================
def _load_database_from_query(
    *,
    query_id: str,
    query_bank_file: Union[str, Path],
    database_path: Union[str, Path],
) -> str:
    """
    Load database DDL context for a given query ID.
    Args:
        - query_id: the ID of the generated query.
        - query_bank_file: the path to the query bank file.
        - database_path: the path to the directory containing the database.
    Returns:
        - A string containing the DDL statements.
    """
    qbank = Path(query_bank_file)
    if not qbank.exists():
        raise FileNotFoundError(f"query_bank not found: {qbank}")

    # 1. Find the record by the given id
    record: Optional[Dict[str, Any]] = None
    with qbank.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("id") == query_id:
                record = obj

    if record is None:
        raise KeyError(f"id '{query_id}' not found in {qbank}")

    database_name = record.get("database_used")
    if not isinstance(database_name, str) or not database_name:
        raise ValueError(
            f"Record for id '{query_id}' has invalid 'database_used': {database_name}"
        )

    # 2. Find the database used
    db_root = Path(database_path)
    if not db_root.exists():
        raise FileNotFoundError(f"path to database does not exist: {db_root}")

    if not database_name.endswith(".sql"):
        database_name = database_name[:-7] + ".sql"

    ddl_matches = sorted(p for p in db_root.rglob(database_name) if p.is_file())

    if not ddl_matches:
        raise FileNotFoundError(f"DDL file not found: '{database_name}' under {db_root}")

    ddl_file_path = ddl_matches[0].resolve()

    # 3. Load DDL content from .sql file
    ddl_content = load_database_ddl(str(ddl_file_path))
    
    return ddl_content


def _generate_content_based_tests(result_table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate basic sanity check tests based solely on the result table structure and content.
    Args:
        result_table: Materialized result table with columns and rows
    Returns:
        List of basic test dictionaries, each containing:
        - description: Human readable description
        - test_type: Type of test (cardinality, data_type, structure, etc.)
        - expected: What we expect to find
        - sql_test: SQL test to execute on the result table
    """
    tests = []

    def _create_test(
        test_type: str, description: str, expected: Any, sql_test: str
    ) -> Dict[str, Any]:
        return {
            "description": description,
            "test_type": test_type,
            "expected": expected,
            "sql_test": sql_test,
        }

    # Extract result data
    result_columns = result_table.get("columns", [])
    result_rows = result_table.get("rows", [])

    # 1. BASIC STRUCTURE TESTS
    # Test that result table has proper structure
    if result_columns:
        tests.append(
            _create_test(
                "structure",
                f"Result table should have exactly {len(result_columns)} columns",
                len(result_columns),
                'SELECT COUNT(*) FROM pragma_table_info("result");',
            )
        )
    else:
        tests.append(
            _create_test(
                "structure",
                "Result table should have at least one column",
                ">=1",
                'SELECT COUNT(*) FROM pragma_table_info("result");',
            )
        )

    # 2. BASIC CARDINALITY TESTS
    # Universal cardinality checks
    if result_rows:
        tests.append(
            _create_test(
                "cardinality",
                f"Result should contain exactly {len(result_rows)} rows",
                len(result_rows),
                'SELECT COUNT(*) FROM "result";',
            )
        )
    else:
        tests.append(
            _create_test(
                "cardinality",
                "Result should contain at least 1 row",
                ">=1",
                'SELECT COUNT(*) FROM "result";',
            )
        )

    # 3. BASIC DATA INTEGRITY TESTS
    # Universal data integrity checks
    if result_rows and result_columns:
        # All-NULL row check (use COALESCE for clarity and performance)
        coalesce_args = ", ".join(f'"{col}"' for col in result_columns)
        tests.append(
            _create_test(
                "data_integrity",
                "Result should not have rows where all columns are NULL",
                "=0",
                f'SELECT COUNT(*) FROM "result" WHERE COALESCE({coalesce_args}) IS NULL;',
            )
        )

    # 4. BASIC UNIQUENESS TESTS
    # Check if result has any duplicate rows
    if result_rows and result_columns and len(result_rows) > 1:
        cols = ", ".join(f'"{col}"' for col in result_columns)
        tests.append(
            _create_test(
                "uniqueness",
                "Check for potential duplicate rows in result",
                f"={len(result_rows)}",
                # Portable: count distinct rows via subquery
                f'SELECT COUNT(*) FROM (SELECT DISTINCT {cols} FROM "result");',
            )
        )

    # 5. FALLBACK TEST
    # Ensure we always have at least one basic test
    if not tests:
        tests.append(
            _create_test(
                "basic_validation",
                "Result table should exist and be accessible",
                ">=0",
                'SELECT COUNT(*) FROM "result";',
            )
        )

    return tests


# ===================================================================
# Forward Pass
# ===================================================================
def generate_groundtruth_forward(
    query_id: str,
    query_bank_file: Union[str, Path],
    database_path: Union[str, Path],
    # LLM call adapter:
    llm_chat: Callable[[List[Dict[str, str]], Optional[int], float], str],
    model: str,
    seed: Optional[int],
    temperature: float,
    out_forward_jsonl: Union[str, Path],
) -> Dict[str, Any]:
    """
    Generate SQL groundtruth from natural language query (forward pass).
    Main pipeline:
        - fetch inputs:  database in DDL format, natural language query, and problem specification
        - prompt LLM for SQL and problem specification
        - validate difficulty score adherence
        - persist JSONL and write <id>.sql
    Returns:
        Dict with groundtruth record containing SQL and difficulty score
    """
    # 1. fetch inputs
    database_ddl = _load_database_from_query(
        query_id=query_id,
        query_bank_file=query_bank_file,
        database_path=database_path,
    )
    query = load_query(query_id=query_id, query_bank_file=query_bank_file)

    # Load query record to get problem specification and database info
    query_record = None
    qbank = Path(query_bank_file)
    with qbank.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("id") == query_id:
                query_record = obj
                break
    if query_record is None:
        raise KeyError(f"Query ID '{query_id}' not found in {query_bank_file}")

    problem_specification = query_record.get("problem_specification")
    database_used = query_record.get("database_used")

    # 2. prompt LLM
    system_prompt = (
        "You are a SQL Generator Agent for a NL-SQL data generation architecture. Given the database, natural-language query, and problem specification, you have the crucial role of crafting the most appropriate, grounded, and accurate SQL to be queried on the provided database that yields an answer to the natural-language query.\n"
        "- The SQL must be executable on the provided database, following proper SQL syntax order: SELECT ... FROM ... [JOIN ...] [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] (no DDL/DML/PRAGMA/ATTACH/etc).\n"
        "- Do not include any natural language text aside from those part of the SQL.\n"
        "- Do not invent non-existent SQL clauses.\n"
        "- Use ONLY tables and columns that exist in the database. Verify all table aliases and column references are correct before generating SQL.\n"
        "- Generate SQL that demonstrates the specified problem family and selected features.\n"
        "- Output ONLY the SQL query wrapped in ``` code blocks, nothing else."
    )

    user_prompt = {
        "task": (
            f"Based on the following database, natural-language query, and problem specification, generate a single SQLite query that answers the natural-language query. "
            "The order of precedence to satisfy in case of logical conflict is: executable on the given database > answers the natural-language query > fulfill the problem family and selected features provided. "
            "Use the problem family and selected features provided to guide your SQL generation. Note that they exist to define the main direction of your query and do not need to completely cover all details."
        ),
        "database": database_ddl,  # Use DDL database context
        "query": query,
        "problem_family": problem_specification.get("problem_family", ""),
        "selected_features": problem_specification.get("selected_features", []),
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]

    raw_message = llm_chat(model, messages, temperature, seed)

    # 3. parse response
    raw_text = raw_message.strip()
    sql_text = ""
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        if len(parts) >= 3:
            # Get the first code block (SQL)
            block = parts[1].strip()
            # Strip potential language tag on first line (e.g., sql, SQL)
            lines = block.splitlines()
            if lines and lines[0].strip().lower() in ("sql", "sqlite"):
                sql_text = "\n".join(lines[1:]).strip()
            else:
                sql_text = block
        else:
            # Fallback: treat everything after ``` as SQL
            sql_text = "```".join(parts[1:]).strip()
    else:
        # Fallback: treat entire response as SQL
        sql_text = raw_text

    if not sql_text:
        raise ValueError("Model must return valid SQL code.")

    # 4. persist artifacts
    record = {
        "id": query_id,
        "difficulty": problem_specification["difficulty"],
        "sql": sql_text,
        "problem_specification": problem_specification,
        "database_used": database_used,
        "model": model,
    }
    persist_result(record, out_forward_jsonl)

    return record


# ===================================================================
# Backward Pass
# ===================================================================
def generate_groundtruth_backward(
    query_id: str,  # original query id
    groundtruth_bank_file: Union[str, Path],
    verdict_path: Union[str, Path],
    query_bank_file: Union[str, Path],
    verdict_forward_file: Union[str, Path],
    out_backward_jsonl: Union[str, Path],
    # LLM call adapter:
    llm_chat: Callable[[List[Dict[str, str]], float, Optional[int]], str],
    model: str,
    temperature: float,
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    Generate fine-grained SQL-based tests from materialized result table using new problem specification from forward verification.
    Main pipeline:
        - Basic tests: Hard-coded universal SQL tests (structure, cardinality, data integrity, and uniqueness)
        - Refined tests: LLM-generated SQL tests tailored to the result table and new problem specification
    Returns:
        {id, difficulty, sql, problem_specification, unit_test, database_used, model}
    """
    # 1. fetch required data
    # Load groundtruth record
    groundtruth_record = load_groundtruth_record(
        query_id=query_id, groundtruth_bank_file=groundtruth_bank_file
    )
    sql = groundtruth_record.get("sql")
    database_used = groundtruth_record.get("database_used")

    if not all([sql, database_used]):
        raise ValueError(
            f"Groundtruth record for query_id '{query_id}' missing required fields: sql or database_used"
        )

    # Load new problem specification from verdict_forward
    verdict_forward_record = None
    verdict_forward_file_obj = Path(verdict_forward_file)
    if verdict_forward_file_obj.exists():
        with verdict_forward_file_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if obj.get("id") == query_id:
                            verdict_forward_record = obj
                            break
                    except Exception:
                        continue

    if verdict_forward_record:
        new_problem_specification = verdict_forward_record.get(
            "problem_specification", {}
        )
        difficulty = new_problem_specification.get(
            "difficulty", groundtruth_record.get("difficulty")
        )
    else:
        # Fallback to original if verdict_forward not found
        new_problem_specification = groundtruth_record.get("problem_specification", {})
        difficulty = groundtruth_record.get("difficulty")

    # Load result table with full data for accurate test generation
    try:
        result_table = load_result_db(
            query_id=query_id,
            result_db_path=verdict_path,
            use_full_data=True,
        )
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Failed to load result table for query_id '{query_id}': {e}")

    # Load the natural language query for full context
    query = load_query(query_id=query_id, query_bank_file=query_bank_file)


    # 2. generate basic content-based tests
    basic_tests = _generate_content_based_tests(result_table)

    # 3. prompt LLM to refine tests
    system_prompt = (
        "You are a SQL unit test validation expert. Given a natural language query, its corresponding SQL query, the result table, and a problem specification, "
        "you have the crucial role of creating a comprehensive SQL-based test suite by:\n"
        "1. CREATE SQL tests that are essential for constructing the result table that answers the natural language query.\n"
        "2. FOCUS on tests that define the uniqueness of the given result table.\n\n"
        "Each test should contain a SQL query that can be executed on the result table (table name: 'result') to validate specific aspects.\n\n"
        "Focus on SQL tests that are essential for answering the natural language query. Consider the query's intent and the new problem specification.\n"
        "Output format: JSON array of test objects:\n"
        "[\n"
        "  {\n"
        '    "description": "Human readable test description",\n'
        '    "test_type": "cardinality|value_existence|data_type|content_validation|ordering|aggregation|filtering|custom",\n'
        '    "expected": "expected value or condition (e.g., >=1, =5, >0)",\n'
        '    "sql_test": "SELECT ... FROM \\"result\\" WHERE ..."\n'
        "  }\n"
        "]\n"
    )

    user_prompt = {
        "task": (
            "Create a list of executable unit test in SQL based on the following information. When designing, consider:\n"
            "- What the natural language query is asking for. The unit test should ensure the corresponding answer in the result table exists.\n"
            "- How the SQL attempts to answer the query. The major changes in the SQL should be reflected in the unit test.\n"
            "- The data type of the result table, as it defines a key property of the answer.\n"
            "- The problem specification, as it defines the general direction of the problem.\n"
        ),
        "query": query,
        "sql": sql,
        "result_table": result_table,
        "problem_specification": new_problem_specification
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]

    try:
        raw_message = llm_chat(model, messages, temperature, seed)

        # Parse LLM response
        raw_text = raw_message.strip()
        if raw_text.startswith("```"):
            # Remove code fences if present
            parts = raw_text.split("```")
            if len(parts) >= 3:
                raw_text = parts[1].strip()
                # Remove language tag if present
                lines = raw_text.splitlines()
                if lines and lines[0].strip().lower() in ["json", "javascript"]:
                    raw_text = "\n".join(lines[1:]).strip()

        unit_test = json.loads(raw_text)

        # Validate and clean unit tests
        refined_tests = []
        if isinstance(unit_test, list):
            for test_obj in unit_test:
                if isinstance(test_obj, dict):
                    required_fields = [
                        "description",
                        "test_type",
                        "expected",
                        "sql_test",
                    ]
                    if all(field in test_obj for field in required_fields):
                        refined_test = {
                            "description": str(test_obj["description"]),
                            "test_type": str(test_obj["test_type"]),
                            "expected": test_obj["expected"],
                            "sql_test": str(test_obj["sql_test"]),
                            "database_used": database_used,
                        }
                        refined_tests.append(refined_test)

        # Fallback: if parsing failed or no valid tests, use basic tests
        if not refined_tests:
            refined_tests = basic_tests.copy()

        unit_test = refined_tests

    except Exception:
        # If LLM fails completely, return basic tests
        unit_test = basic_tests.copy()

    # 7. Persist artifacts
    record = {
        "id": query_id,
        "difficulty": difficulty,
        "sql": sql,
        "problem_specification": new_problem_specification,
        "unit_test": unit_test,
        "database_used": database_used,
        "model": model,
    }
    persist_result(record, out_backward_jsonl)

    return record
