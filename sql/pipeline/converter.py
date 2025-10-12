"""
Converter for parallel batch generation results to HuggingFace dataset format.

This module aggregates results from parallel batch iterations and converts them
into a structured dataset format for training.

Features:
- Aggregates batch-indexed files from parallel generation (e.g., query_forward_0.jsonl, query_forward_1.jsonl)
- Only includes successfully verified records (verdict: correct + adherence: adheres)
- Combines forward pass SQL (solution) with backward pass query (message) and groundtruth

Output format:
- id: Unique identifier for the query/SQL pair (same across forward/backward)
- message: {"content": "natural language query", "role": "user"}
- groundtruth: Unit test specifications from backward pass
- solution: SQL statement from forward pass
- difficulty: Difficulty level (easy/medium/hard)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union

sys.path.append(str(Path(__file__).parent.parent))


# ===================================================================
# Utility Functions
# ===================================================================

def _load_jsonl_by_id(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load a .jsonl file and return a dict id -> record, skips malformed lines.
    """
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                id = obj.get("id")
                if isinstance(id, str) and id:
                    out[id] = obj
            except Exception:
                continue
    return out


def _extract_message_and_difficulty(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch query, difficulty, and database used from query_backward.
    """
    query = str(record.get("query", "")).strip()
    difficulty = record.get("difficulty", "")
    return {
        "message": {"content": query, "role": "user"},
        "difficulty": difficulty
    }


def _extract_groundtruth(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch groundtruth unit tests from groundtruth_backward.
    """
    groundtruth = record.get("unit_test", [])
    return {
        "groundtruth": groundtruth
    }


def _extract_solution(query_id: str, verdict_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Fetch SQL as formatted string from .sql file in verdict path.
    """
    sql = ""
    if query_id and verdict_path:
        sql_file = Path(verdict_path) / f"{query_id}.sql"
        if sql_file.exists():
            sql = sql_file.read_text(encoding="utf-8").strip()
    return {"solution": sql}


def _is_successful(record: Dict[str, Any]) -> bool:
    """
    Check if a verdict was successful based on verdict results.
    Args:
        verdict_record: Verdict record from any verdict file
    Returns:
        True if the verdict was successful (correct verdict and adherence is adheres or partial)
    """
    verdict = record.get("verdict", "").lower()
    adherence = record.get("adherence", "").lower()
    # Success criteria: verdict must be correct and adherence must be adheres or partial
    return verdict == "correct" and adherence in ["adheres", "partial"]


# ===================================================================
# Main Pipeline
# ===================================================================
def convert(
    *,
    result_base_path: Union[str, Path],
    out_jsonl: Union[str, Path],
    drop_incomplete: bool,
) -> Dict[str, Any]:
    """
    Convert parallel batch generation results to HuggingFace dataset format.
    Logic:
    - Aggregates all batch-indexed files from parallel generation iterations
    - Only includes records where final verification (step 7) was successful
    - Combines forward pass SQL (solution) with backward pass query (message) and groundtruth
    - Drops incomplete records if drop_incomplete=True
    """
    result_base = Path(result_base_path)

    # Load all batch-indexed files from parallel generation
    backward_queries = {}
    backward_groundtruths = {}
    backward_verdicts = {}

    query_dir = result_base / "query"
    groundtruth_dir = result_base / "groundtruth"
    verdict_dir = result_base / "verdict"

    # Load consolidated backward pass files (these contain the data we need)
    if query_dir.exists():
        backward_file = query_dir / "query_backward.jsonl"
        if backward_file.exists():
            backward_queries.update(_load_jsonl_by_id(backward_file))

    if groundtruth_dir.exists():
        backward_file = groundtruth_dir / "groundtruth_backward.jsonl"
        if backward_file.exists():
            backward_groundtruths.update(_load_jsonl_by_id(backward_file))

    if verdict_dir.exists():
        backward_file = verdict_dir / "verdict_backward.jsonl"
        if backward_file.exists():
            backward_verdicts.update(_load_jsonl_by_id(backward_file))

    # Get all query IDs from backward queries
    all_query_ids = set(backward_queries.keys())

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    successful = 0
    failed = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for qid in all_query_ids:
            has_complete_data = (
                qid in backward_queries 
                and qid in backward_groundtruths 
                and qid in backward_verdicts
            )

            if not has_complete_data:
                if drop_incomplete:
                    skipped += 1
                    continue

            # Check if final verification was successful
            verification_successful = False
            if qid in backward_verdicts:
                verdict_record = backward_verdicts[qid]
                verification_successful = _is_successful(verdict_record)

            if not verification_successful:
                failed += 1
                if drop_incomplete:
                    skipped += 1
                    continue
            else:
                successful += 1

            # Extract data using helper functions
            query_record = backward_queries.get(qid, {})
            groundtruth_record = backward_groundtruths.get(qid, {})

            message_data = _extract_message_and_difficulty(query_record)
            groundtruth_data = _extract_groundtruth(groundtruth_record)
            solution_data = _extract_solution(qid, verdict_dir)

            # Create output record
            out_obj = {
                "id": qid,
                "message": message_data["message"],
                "ground_truth": groundtruth_data["groundtruth"],
                "solution": solution_data["solution"],
                "dataset": "sql",
                "difficulty": message_data["difficulty"]
            }

            # Skip incomplete records if requested
            if drop_incomplete:
                if (
                    not out_obj["solution"]
                    or not out_obj["message"]["content"]
                    or not out_obj["ground_truth"]
                ):
                    skipped += 1
                    continue

            # Ensure minimal fields exist for incomplete records
            if not drop_incomplete:
                if not out_obj["message"]["content"]:
                    out_obj["message"] = {"content": "", "role": "user"}
                if not out_obj["solution"]:
                    out_obj["solution"] = ""
                if not out_obj["ground_truth"]:
                    out_obj["ground_truth"] = []

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    return {
        "result_base_path": str(result_base_path),
        "output_jsonl": str(out_jsonl),
        "written": written,
        "skipped": skipped,
        "drop_incomplete": drop_incomplete,
        "total_query_ids": len(all_query_ids),
        "verification_results": {
            "successful": successful,
            "failed": failed,
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert pipeline results into a single dataset.jsonl."
    )
    ap.add_argument(
        "--result_base_path",
        default="./result",
        help="Base path containing query/, groundtruth/, and verdict/ directories",
    )
    ap.add_argument(
        "--out_jsonl", default="./result/dataset.jsonl", help="Output dataset file path"
    )
    ap.add_argument(
        "--drop_incomplete",
        type=bool,
        default=True,
        help="True = skip incomplete records; False = include with blanks",
    )
    args = ap.parse_args()

    out = convert(
        result_base_path=args.result_base_path,
        out_jsonl=args.out_jsonl,
        drop_incomplete=args.drop_incomplete,
    )
    print(json.dumps(out, ensure_ascii=False))
