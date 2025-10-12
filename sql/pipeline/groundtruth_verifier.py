import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import (
    persist_result,
    load_sql,
    load_query,
    load_result_db,
    load_groundtruth_record,
    load_problem_family,
    calculate_difficulty,
    get_difficulty_range
)


# ===================================================================
# Utility Functions
# ===================================================================
def evaluate_test_result(actual_value: Any, expected: str) -> bool:
    """
    Evaluate if actual test result matches expected value.

    Args:
        actual_value: The actual result from executing SQL test
        expected: Expected value string (may contain operators like >=1, =0, >5)

    Returns:
        True if test passes, False otherwise
    """
    if actual_value is None:
        return expected.lower() in ["null", "none", "=0", "0"]

    expected_str = str(expected).strip()

    # Handle comparison operators
    if expected_str.startswith(">="):
        try:
            threshold = float(expected_str[2:])
            return float(actual_value) >= threshold
        except (ValueError, TypeError):
            return False
    elif expected_str.startswith("<="):
        try:
            threshold = float(expected_str[2:])
            return float(actual_value) <= threshold
        except (ValueError, TypeError):
            return False
    elif expected_str.startswith(">"):
        try:
            threshold = float(expected_str[1:])
            return float(actual_value) > threshold
        except (ValueError, TypeError):
            return False
    elif expected_str.startswith("<"):
        try:
            threshold = float(expected_str[1:])
            return float(actual_value) < threshold
        except (ValueError, TypeError):
            return False
    elif expected_str.startswith("="):
        try:
            expected_val = float(expected_str[1:])
            return float(actual_value) == expected_val
        except (ValueError, TypeError):
            return str(actual_value) == expected_str[1:]
    else:
        # Direct comparison
        try:
            return float(actual_value) == float(expected_str)
        except (ValueError, TypeError):
            return str(actual_value) == expected_str


def _calculate_sql_difficulty_score(
    problem_specification: Dict[str, Any], problem_family_config: Dict[str, Any]
) -> float:
    """
    Calculate the difficulty score based on problem specification using problem family configuration.

    Args:
        problem_specification: Dict containing problem_family and selected_features
        problem_family_config: The loaded problem family configuration

    Returns:
        The calculated difficulty score
    """
    problem_family = problem_specification.get("problem_family", "")
    selected_features = problem_specification.get("selected_features", [])

    if not isinstance(selected_features, list):
        selected_features = []

    # Calculate total difficulty score
    total_score = 0.0

    # Add problem family base score
    problem_family_base_scores = problem_family_config.get("problem_family_base", {})
    if problem_family in problem_family_base_scores:
        base_score = problem_family_base_scores[problem_family]
        total_score += base_score

    # Add selected feature scores
    family_features = problem_family_config.get("family_features", {}).get(problem_family, {})
    for feature_name in selected_features:
        if feature_name in family_features:
            feature_score = family_features[feature_name].get("difficulty_score", 0.0)
            total_score += feature_score

    return max(0.0, total_score)  # Ensure non-negative score


def check_difficulty_adherence(
    calculated_score: float, target_difficulty: str, problem_family: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if a calculated difficulty score adheres to the target difficulty range.

    Args:
        calculated_score: The calculated difficulty score
        target_difficulty: The target difficulty level
        problem_family: The loaded problem family configuration

    Returns:
        Dict with adherence result and details
    """
    try:
        # Get difficulty range directly from problem family configuration
        difficulty_levels = problem_family.get("difficulty", {}).get("levels", {})
        target_difficulty_norm = target_difficulty.strip().lower()
        
        if target_difficulty_norm not in difficulty_levels:
            available = ", ".join(sorted(difficulty_levels.keys()))
            raise ValueError(f"Unknown difficulty '{target_difficulty}'. Available: {available}")
        
        difficulty_range = difficulty_levels[target_difficulty_norm]
        min_score = difficulty_range["min"]
        max_score = difficulty_range["max"]

        if min_score <= calculated_score <= max_score:
            adherence = "adheres"
            justification = f"Score {calculated_score:.2f} is within {target_difficulty} range [{min_score}-{max_score}]"
        else:
            adherence = "violates"
            if calculated_score < min_score:
                justification = f"Score {calculated_score:.2f} is below {target_difficulty} range [{min_score}-{max_score}] (too easy)"
            else:
                justification = f"Score {calculated_score:.2f} is above {target_difficulty} range [{min_score}-{max_score}] (too hard)"

        return {
            "adherence": adherence,
            "calculated_score": calculated_score,
            "target_range": {"min": min_score, "max": max_score},
            "target_difficulty": target_difficulty,
            "justification": justification,
        }

    except ValueError as e:
        return {
            "adherence": "partial",
            "calculated_score": calculated_score,
            "target_range": None,
            "target_difficulty": target_difficulty,
            "justification": f"Error checking adherence: {str(e)}",
        }


# ===================================================================
# Forward Pass
# ===================================================================
def verify_groundtruth_forward(
    query_id: str,
    query_bank_path: Union[str, Path],
    result_db_path: Union[str, Path],
    groundtruth_bank_path: Union[str, Path],
    problem_family_file: Union[str, Path],
    # LLM call adapter:
    llm_chat: Callable[[List[Dict[str, Any]], float, Optional[int]], str],
    model: str,
    temperature: float,
    seed: Optional[int],
    out_jsonl: Union[str, Path],
) -> Dict[str, Any]:
    """
    Main pipeline for forward pass verification:
        - Check adherence: LLM analyzes SQL complexity/task_type, compare with original specification
        - Check correctness: result table vs query (offer revision if incorrect)
    """
    # 1. fetch inputs
    query = load_query(query_id=query_id, query_bank_file=query_bank_path)
    result_table = load_result_db(
        query_id=query_id,
        result_db_path=result_db_path,
        use_full_data=True,
    )
    sql = load_sql(query_id=query_id, groundtruth_bank_file=groundtruth_bank_path)
    groundtruth_record = load_groundtruth_record(
        query_id=query_id, groundtruth_bank_file=groundtruth_bank_path
    )
    problem_family = load_problem_family(problem_family_file)
    original_problem_specification = groundtruth_record.get("problem_specification", {})

    # 2. prompt LLM to determine adherence and correctness
    system_prompt = (
        "You are a SQL Verification Agent for a NL-SQL data generation architecture. You have two critical tasks:\n"
        "\n"
        "TASK 1 - ADHERENCE ANALYSIS:\n"
        "Analyze the provided SQL query and identify which specific features from the assigned problem family it demonstrates. "
        "Look at the SQL structure, clauses, operations, and logic to determine which features from the given problem family the SQL actually exhibits. "
        "The problem family is already assigned and should not be changed - only identify which features within that family are present.\n"
        "TASK 2 - CORRECTNESS ANALYSIS:\n"
        "Determine if the result table correctly answers the natural language query. "
        "Analyze the correlation between query context and result table columns/values. "
        "Report 'correct' if the result can answer the query, 'partial' if can't tell but the result make sense (i.e. it provides useful information not necessarily for the query provided), and 'incorrect' if the result data type does not match the data type of the expected answer.\n"
        "\n"
        "Output your analysis in this format:\n"
        "{\n"
        '  "adherence_analysis": {\n'
        '    "selected_features": ["feature1", "feature2", ...]  // Names from the assigned family features that are actually present in the SQL\n'
        "  },\n"
        '  "correctness_analysis": {\n'
        '    "verdict": "correct" | "incorrect" | "partial",\n'
        '    "critique": "detailed feedback about correctness"\n'
        "  }\n"
        "}\n"
        "\n"
    )

    # Extract the assigned problem family from original specification
    assigned_problem_family = original_problem_specification.get("problem_family", "")
    
    user_prompt = {
        "task": "Analyze the SQL for adherence patterns and verify if the result table correctly answers the natural language query. Use your best judgement. Do not create any new features. Focus on identifying which features from the assigned problem family are actually present in the SQL. For correctness analysis, your focus should be on the correlation between the query and the results. Consider data type and context of the problem to aid your judgement.",
        "query": query,
        "sql": sql,
        "result_table": result_table,
        "assigned_problem_family": assigned_problem_family,
        "available_features": problem_family.get("family_features", {}).get(assigned_problem_family, {}),
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]

    raw_response = llm_chat(
        model=model,
        messages=messages,
        temperature=temperature,
        seed=seed,
    )

    # 3. parse response
    verdict = "partial"
    critique = ""
    new_problem_specification = {}
    adherence = "partial"

    try:
        response_data = json.loads(raw_response.strip())

        # Extract correctness analysis
        correctness_analysis = response_data.get("correctness_analysis", {})
        verdict = str(correctness_analysis.get("verdict", "partial")).lower().strip()
        if verdict not in {"correct", "incorrect", "partial"}:
            verdict = "partial"
        critique = str(correctness_analysis.get("critique", "")).strip()

        # Extract adherence analysis and calculate difficulty
        adherence_analysis = response_data.get("adherence_analysis", {})
        if adherence_analysis:
            # Preserve the original problem family, only update selected features
            updated_specification = {
                "problem_family": assigned_problem_family,
                "selected_features": adherence_analysis.get("selected_features", [])
            }
            new_problem_specification = calculate_difficulty(
                updated_specification, problem_family
            )

            # Compare with original problem specification for adherence
            def _normalize_list(str_list):
                return set(s.strip().lower() for s in str_list if isinstance(s, str))

            original_difficulty = (
                str(original_problem_specification.get("difficulty", ""))
                .strip()
                .lower()
            )
            original_features = _normalize_list(
                original_problem_specification.get("selected_features", [])
            )

            new_difficulty = (
                str(new_problem_specification.get("difficulty", "")).strip().lower()
            )
            new_features = _normalize_list(
                new_problem_specification.get("selected_features", [])
            )

            difficulty_match = original_difficulty == new_difficulty
            features_match = original_features == new_features

            if difficulty_match and features_match:
                adherence = "adhere"
            elif difficulty_match or features_match:
                adherence = "partial"
            else:
                adherence = "violates"
        else:
            adherence = "partial"
            new_problem_specification = original_problem_specification

    except Exception as e:
        critique = f"Failed to parse LLM response: {str(e)}"
        adherence = "partial"
        new_problem_specification = original_problem_specification

    # 4. Create output
    out = {
        "id": query_id,
        "verdict": verdict,
        "adherence": adherence,
        "critique": critique,
        "problem_specification": new_problem_specification,
        "model": model,
    }

    persist_result(out, out_jsonl)
    return out


# ===================================================================
# Backward Pass
# ===================================================================
def verify_groundtruth_backward(
    query_id: str,
    result_db_path: Union[str, Path],
    groundtruth_bank_path: Union[str, Path],
    problem_family_file: Union[str, Path],
    out_jsonl: Union[str, Path],
) -> Dict[str, Any]:
    """
    Backward pass verification pipeline:
        - Check correctness: Execute unit test SQL queries on result table
        - Check adherence: Check difficulty score against problem specification
    Args:
        query_id: The query ID to verify
        result_db_path: Path to verdict directory containing result .sqlite files
        groundtruth_bank_path: Path to backward groundtruth bank file
        problem_family_file: Path to problem_family.json for difficulty scoring
        out_jsonl: Output file path for verdict
    Returns:
        Dict with verification results (verdict, adherence, etc.)
    """
    # 1. Load backward groundtruth record to get unit tests
    backward_groundtruth = load_groundtruth_record(
        query_id=query_id, groundtruth_bank_file=groundtruth_bank_path
    )

    unit_tests = backward_groundtruth.get("unit_test", [])
    if not unit_tests:
        out = {
            "id": query_id,
            "verdict": "partial",
            "adherence": "partial",
            "justification": "No unit tests found in backward groundtruth",
            "critique": "Unable to verify correctness without unit tests. The backward groundtruth generation may have failed.",
        }
        persist_result(out, out_jsonl)
        return out

    # 2. Execute unit tests on result table to determine correctness
    db_path = Path(result_db_path) / f"{query_id}.sqlite"
    if not db_path.exists():
        out = {
            "id": query_id,
            "verdict": "partial",
            "adherence": "partial",
            "justification": "Result database not found",
            "critique": f"Cannot execute unit tests: result database {db_path} does not exist.",
        }
        persist_result(out, out_jsonl)
        return out

    # Execute unit tests
    test_results = []
    passed_tests = 0
    total_tests = 0

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        for test in unit_tests:
            test_sql = test.get("sql_test", "")
            expected = test.get("expected", "")
            description = test.get("description", "Unknown test")

            if not test_sql:
                continue

            total_tests += 1

            try:
                # Execute the test SQL
                cur.execute(test_sql)
                result = cur.fetchone()
                actual_value = result[0] if result else None

                # Compare with expected value
                test_passed = evaluate_test_result(actual_value, expected)

                test_results.append(
                    {
                        "description": description,
                        "sql_test": test_sql,
                        "expected": expected,
                        "actual": actual_value,
                        "passed": test_passed,
                    }
                )

                if test_passed:
                    passed_tests += 1

            except Exception as e:
                test_results.append(
                    {
                        "description": description,
                        "sql_test": test_sql,
                        "expected": expected,
                        "actual": None,
                        "passed": False,
                        "error": str(e),
                    }
                )

    except Exception as e:
        out = {
            "id": query_id,
            "verdict": "partial",
            "adherence": "partial",
            "justification": f"Database connection error: {str(e)}",
            "critique": "Unable to execute unit tests due to database connection issues.",
        }
        persist_result(out, out_jsonl)
        return out
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    # 3. Determine correctness based on test results (hard-coded logic)
    if total_tests == 0:
        correctness_score = 1.0  # No tests to fail
    else:
        correctness_score = passed_tests / total_tests

    # Hard-coded verdict based on unit test results
    if correctness_score == 1.0:  # All tests passed
        verdict = "correct"
    elif correctness_score > 0.0:  # Some tests passed
        verdict = "partial"
    else:  # No tests passed
        verdict = "incorrect"

    # 4. Check adherence based on difficulty score against problem specification
    adherence = "partial"  # Default
    adherence_justification = ""

    if verdict == "correct":
        # Load difficulty and problem specification from backward groundtruth for adherence check
        difficulty = backward_groundtruth.get("difficulty", "")
        problem_specification = backward_groundtruth.get("problem_specification", {})

        # Check if the original SQL (from groundtruth) meets the target difficulty
        if problem_specification and problem_specification.get("target_score_range"):
            original_sql = backward_groundtruth.get("sql", "")
            if original_sql:
                try:
                    # Load problem family for scoring
                    problem_family = load_problem_family(problem_family_file)
                    sql_score = _calculate_sql_difficulty_score(
                        problem_specification, problem_family
                    )
                    difficulty_check = check_difficulty_adherence(
                        sql_score, difficulty, problem_family
                    )
                    adherence = difficulty_check["adherence"]
                    adherence_justification = difficulty_check["justification"]
                except Exception as e:
                    adherence = "partial"
                    adherence_justification = (
                        f"Error during difficulty verification: {str(e)}"
                    )
            else:
                adherence = "partial"
                adherence_justification = (
                    "No SQL found in backward groundtruth for difficulty verification"
                )
        else:
            adherence = "partial"
            adherence_justification = (
                "Missing problem specification for difficulty verification"
            )
    else:
        # Don't check adherence if correctness is not "correct"
        adherence_justification = "Adherence not checked - correctness is not 'correct'"

    # 5. Create final justification combining test results and adherence feedback
    test_summary = f"Unit tests: {passed_tests}/{total_tests} passed (score: {correctness_score:.2f})"

    if adherence_justification:
        final_justification = f"{test_summary}. {adherence_justification}"
    else:
        final_justification = test_summary

    # 6. Create output
    out = {
        "id": query_id,
        "verdict": verdict,
        "adherence": adherence,
        "justification": final_justification,
        "test_results": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "correctness_score": correctness_score,
            "details": test_results,
        },
    }

    # Always write to verdict_backward.jsonl
    persist_result(out, out_jsonl)

    return out
