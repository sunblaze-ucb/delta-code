import json
import pathlib
import random
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import (
    persist_result,
    load_groundtruth_record,
    load_database_ddl,
    load_problem_family,
    get_difficulty_range,
    calculate_difficulty,
)


# ===================================================================
# Utility Functions
# ===================================================================
def _load_query_style(
    query_style_file: Union[str, Path],
    seed: Optional[int],
) -> List[str]:
    """
    Read 'query_style.json' (dict of lists) and select <= 1 phrase per randomly chosen group.
        - Step 1 (groups): randomly select a subset of groups.
        - Step 2 (phrases): for each selected group, pick one random phrase from that group.
    Returns: [<phrases>]
    """
    rng = random.Random(seed)
    obj = json.loads(Path(query_style_file).read_text(encoding="utf-8"))
    available_sets = list(obj.keys())

    # 1. select groups
    k = rng.randint(0, len(available_sets))  # can be 0 .. len
    selected_groups = rng.sample(available_sets, k)

    # 2. pick one phrase per selected group
    phrases: List[str] = []
    for word in selected_groups:
        choice = rng.choice(obj[word])
        phrases.append(choice)

    return phrases


def _format_complete_query(natural_query: str, database_ddl: str) -> str:
    """
    Format a complete query with database DDL context and return format specifications.
    Args:
        natural_query: The original natural language query/question
        database_ddl: The DDL schema context for the database
    Returns:
        A formatted query string with complete context
    """
    formatted_query = f"""
    {natural_query.strip()} Return your answer strictly in the following format: ```SQL```. Do not include any other text irrelevant to the SQL itself. Here is the database schema representing the source that your response will be queried on. Note that the row values are truncated and do not represent the entire table:
    ```sql
    {database_ddl.strip()}
    ```
    """
    return formatted_query


# ===================================================================
# Forward Pass
# ===================================================================
def generate_query_forward(
    difficulty: str,
    database_path: Union[str, Path],
    query_style_file: Union[str, Path],
    problem_family_file: Union[str, Path],
    guided_spec: Dict[str, Any],
    # LLM call adapter:
    llm_chat: Callable[[List[Dict[str, str]], Optional[int], float], str],
    model: str,
    temperature: float,
    seed: Optional[int],
    out_forward_jsonl: Union[str, Path],
) -> Dict[str, Any]:
    """
    Generate natural language query from database schema and settings (forward pass).
    Main pipeline:
        - load inputs: database in DDL format, query style, problem specification
        - prompt LLM for natural language query and chosen problem specification
        - parse response into standardized format
        - persist artifacts to a query bank
    Returns:
        dictionary: {id, difficulty, query, problem_specification, database_used, model}
    """
    rng = random.Random(seed)
    db_root = Path(database_path)

    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be 'easy', 'medium', or 'hard'.")

    # 1. Load inputs
    candidates = sorted([p for p in db_root.rglob("*.sql") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No .sql files found under: {db_root}")
    database_path = rng.choice(candidates)

    database_ddl = load_database_ddl(database_path)
    query_style = _load_query_style(
        query_style_file,
        seed=seed,
    )
    target_score_range = get_difficulty_range(problem_family_file, difficulty)

    # Use guided specification (required) - contains problem family and selected features
    if not guided_spec:
        raise ValueError("guided_spec is required and must contain problem family with selected features")
    
    # Extract problem family name and features from guided_spec
    if not isinstance(guided_spec, dict) or len(guided_spec) != 1:
        raise ValueError("guided_spec must be a dict with exactly one problem family")
    
    problem_family_name = list(guided_spec.keys())[0]
    selected_features = guided_spec[problem_family_name]

    # 2. prompt LLM
    system_prompt = (
        "You are a Natural Language Query Generator Agent for a NL-SQL data generation architecture. Given the database, query style, and problem specification, you have the crucial role of crafting the most appropriate, grounded, and diversified task queries.\n"
        "- The query must be realistically answerable from solely the database.\n"
        "- Do NOT include any SQL code or answer to the query.\n"
        "- Do NOT invent none-existent data values.\n"
        "- Do NOT ask questions beyond what can be answered solely from the database.\n"
        "- Generate a query that will result in SQL with difficulty score within the target range.\n"
        "- Use the provided problem family and selected features to guide your query generation. These define the main direction and complexity patterns your query should exhibit. You are not required to use all the features provided if they do not fit to the context of the database and thus query.\n"
        "- Output ONLY a strict JSON: {'query': <natural language query text>, 'problem_specification': {'problem_family': <problem family name>, 'selected_features': [<feature names used>]}}."
    )

    user_prompt = {
        "task": (
            "Based on the following database, query style, and problem specification, generate 1 natural language query of the specified difficulty. "
            "Your generated natural language query must be answerable by querying on the database given. "
            "Utilize the information provided below to generate the query. The order of precedence to satisfy in case of logical conflict is: answerable > within target difficulty score range > adhering to problem family and features > fulfilling query style. "
            "Use the provided problem family and selected features to guide your query generation. "
            "Your adopted problem specification should have a total difficulty score within the target range."
        ),
        "database": database_ddl,  # DDL schema from load_database_ddl
        "difficulty": difficulty,
        "target_score_range": target_score_range,
        "problem_family": problem_family_name,
        "selected_features": selected_features,
        "style": query_style
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]

    raw_message = llm_chat(model, messages, temperature, seed)

    # 3. parse response
    raw_text = raw_message.strip()
    if raw_text.startswith("```"):
        # Remove the first line (``` or ```json)
        lines = raw_text.splitlines()
        # Find the first line that does not start with ```
        start = 0
        while start < len(lines) and lines[start].strip().startswith("```"):
            start += 1
        # Find the last line that is not a closing ```
        end = len(lines)
        while end > start and lines[end - 1].strip().startswith("```"):
            end -= 1
        raw_text = "\n".join(lines[start:end]).strip()
    try:
        response_data = json.loads(raw_text)
        query_text = response_data.get("query", "").strip()
        llm_problem_specification = response_data.get("problem_specification", {})
        if not query_text:
            raise ValueError("Model returned no query.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}")

    query = [query_text]

    # 4. persist artifacts
    records = []
    for q in query:
        # Format complete query with DDL context and return format
        complete_query = _format_complete_query(
            natural_query=q, database_ddl=database_ddl
        )

        # Load problem family to calculate actual difficulty based on LLM choices
        problem_family = load_problem_family(problem_family_file)
        calculated_spec = calculate_difficulty(
            llm_problem_specification, problem_family
        )

        record = {
            "id": str(uuid.uuid4()),
            "difficulty": calculated_spec["difficulty"],
            "query": complete_query,
            "problem_specification": calculated_spec,
            "database_used": database_path.name,
            "model": model,
        }
        persist_result(record, out_forward_jsonl)
        records.append(record)

    return records


# ===================================================================
# Backward Pass
# ===================================================================
def generate_query_backward(
    query_id: str,
    groundtruth_bank_file: Union[str, Path],
    verdict_path: Union[str, Path],
    database_path: Union[str, Path],
    verdict_forward_file: Union[str, Path],
    # LLM call adapter:
    llm_chat: Callable[[List[Dict[str, str]], Optional[int], float], str],
    model: str,
    temperature: float,
    seed: Optional[int],
    out_backward_jsonl: Union[str, Path],
) -> Dict[str, Any]:
    """
    Generate natural language query from (validated) SQL, (new) problem specification, and database context (backward pass).
    Main pipeline:
        - Load inputs: SQL, problem specification, and database context
        - Prompt LLM for natural language query
        - Parse response into standardized format
        - Persist artifacts to a query bank
    Returns:
        {id, difficulty, query, problem_specification, database_used, model}
    """
    # 1. load inputs
    # Load validated SQL from verdict directory
    validated_sql_path = Path(verdict_path) / f"{query_id}.sql"
    if not validated_sql_path.exists():
        raise FileNotFoundError(f"Validated SQL file not found: {validated_sql_path}")
    validated_sql = validated_sql_path.read_text(encoding="utf-8").strip()

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

    if not verdict_forward_record:
        raise ValueError(f"verdict_forward_record not found for query_id '{query_id}'")
    problem_specification = verdict_forward_record.get("problem_specification", {})
    difficulty = problem_specification.get("difficulty")

    # Load database context
    groundtruth_record = load_groundtruth_record(
        query_id=query_id, groundtruth_bank_file=groundtruth_bank_file
    )
    database_used = groundtruth_record.get("database_used")

    db_root = Path(database_path)
    if not database_used.endswith(".sql"):
        database_used = database_used[:-7] + ".sql"
    ddl_matches = sorted(p for p in db_root.rglob(database_used) if p.is_file())
    if not ddl_matches:
        raise FileNotFoundError(
            f"DDL file not found: '{database_used}' under {db_root}"
        )

    # Load database DDL for accurate query generation
    database_ddl = load_database_ddl(str(ddl_matches[0]))

    if not all([validated_sql, difficulty, database_used]):
        raise ValueError(
            f"Missing required fields: validated_sql, difficulty, or database_used"
        )

    # 2. prompt LLM
    system_prompt = (
        "You are a Natural Language Query Generator Agent for a SQL-NL data generation architecture. Given the SQL, problem specification (problem family and selected features), and database context, you have the crucial role of crafting the most appropriate, grounded, and accurate task query.\n"
        "- The query must be answerable by the provided SQL querying on the given database.\n"
        "- Do NOT include any SQL code or answer to the query.\n"
        "- Do NOT invent none-existent data values.\n"
        "- Do NOT ask questions beyond what can be answered solely from the database.\n"
        "- Generate a query that will result in SQL with difficulty score within the target range.\n"
        "- Utilize the problem family and selected features provided in the problem_specification.\n"
        "- Output ONLY the natural language query, nothing else."
    )

    user_prompt = {
        "task": (
            "Based on the following information, generate a natural language query that can be answered by the provided SQL querying on the given database. In other words, generate a question based on the following answer (SQL), while adhering to the problem family and selected features specified. "
            "Your generated natural language query must be answerable by querying on the database given. "
            "Utilize the information provided. The order of precedence to satisfy in case of logical conflict is: answerable by the result_table > adhering to the context of the database > satisfying problem_specification. "
            "Utilize the problem family and selected features provided in the problem_specification."
        ),
        "sql": validated_sql,
        "database": database_ddl,
        "problem_specification": problem_specification
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]

    raw_message = llm_chat(model, messages, temperature, seed)

    # 3. parse response
    query_text = raw_message.strip()
    if not query_text:
        raise ValueError("Model returned no query.")
    complete_query = _format_complete_query(
        natural_query=query_text, database_ddl=database_ddl
    )
    database_used = database_used.replace(".sqlite", ".sql")
    
    # 4. persist artifacts
    record = {
        "id": query_id,
        "difficulty": difficulty,
        "query": complete_query,
        "problem_specification": problem_specification,
        "database_used": database_used,
        "model": model,
    }

    persist_result(record, out_backward_jsonl)

    return record
