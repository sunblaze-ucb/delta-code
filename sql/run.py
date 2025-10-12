#!/usr/bin/env python3
"""
Pipeline orchestrator for SQL generation with 9-step workflow.

9-Step Workflow:
1. Setup sampler based on batch_size, num_iteration, and difficulty
2. Generate query (forward)
3. Generate groundtruth (forward)
4. Verify format
5. Verify groundtruth (forward)
6. Generate unit test (backward)
7. Generate query based on groundtruth (backward)
8. Verify again (backward)
9. Save to dataset.jsonl if correct and adhere
"""

import argparse
import json
import sys
import yaml
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add current directory to path for pipeline imports
sys.path.append(str(Path(__file__).parent))

from pipeline.query_generator import generate_query_forward, generate_query_backward
from pipeline.groundtruth_generator import generate_groundtruth_forward, generate_groundtruth_backward
from pipeline.format_verifier import verify_format
from pipeline.groundtruth_verifier import verify_groundtruth_forward, verify_groundtruth_backward
from pipeline.converter import convert
from sampler import create_specification_subsets
from utils.utils import llm_chat


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_config_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and map configuration parameters to pipeline function arguments."""
    # Get configuration sections
    paths = config.get("paths", {})
    pipeline_config = config.get("pipeline", {})
    execution_config = config.get("execution", {})
    temp_config = config.get("temperature", {})
    conv_config = config.get("converter", {})
    
    # Map shared/model config
    model_config = config.get("model", {})
    if isinstance(model_config, dict):
        model_name = model_config.get("name", "o4-mini")
    
    # Extract parameters
    params = {
        # Core pipeline parameters
        "difficulty": pipeline_config.get("difficulty"),
        
        # Path parameters
        "database_path": paths.get("database_path"),
        "query_style_file": paths.get("query_style_file"),
        "problem_family_file": paths.get("problem_family_file"),
        "result_base_path": paths.get("result_base"),
        
        # Model parameters
        "model": model_name,
        "seed": config.get("seed"),
        
        # Execution parameters
        "batch_size": execution_config.get("batch_size"),
        "num_iteration": execution_config.get("num_iteration"),
        "diversity_sampling": execution_config.get("diversity_sampling"),
        "sampling_tolerance": execution_config.get("tolerance"),
        "single_problem_family": execution_config.get("single_problem_family"),
        
        # Temperature parameters
        "query_generation_forward_temperature": temp_config.get("query_generation_forward"),
        "groundtruth_generation_temperature": temp_config.get("groundtruth_generation"),
        "groundtruth_verification_temperature": temp_config.get("groundtruth_verification"),
        "query_generation_backward_temperature": temp_config.get("query_generation_backward"),
        
        # Converter parameters
        "drop_incomplete": conv_config.get("drop_incomplete"),
        
        # Other parameters
        "verbose": config.get("verbose"),
    }

    return params


def run_single_pipeline(
    pipeline_id: str,
    guided_spec: Optional[Dict[str, Any]],
    params: Dict[str, Any],
    result_base_path: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run a single 9-step pipeline instance."""
    start_time = time.time()
    
    # Create result directories directly under result base path
    result_dir = result_base_path
    result_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["query", "groundtruth", "verdict"]:
        (result_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate guided_spec is properly provided by sampler
        if not guided_spec:
            raise ValueError("guided_spec is None - sampler failed to provide specification subset")
        
        # New format: {problem_family_name: {feature_name: feature_config, ...}}
        if not isinstance(guided_spec, dict) or len(guided_spec) != 1:
            raise ValueError(f"guided_spec must be a dict with exactly one problem family. Got: {guided_spec}")
        
        problem_family_name = list(guided_spec.keys())[0]
        family_features = guided_spec[problem_family_name]
        if not isinstance(family_features, dict):
            raise ValueError(f"Problem family '{problem_family_name}' must have dict of features. Got: {type(family_features)}")
        
        # Step 2: Generate query (forward)
        if verbose:
            print(f"   🧠 Generating query for pipeline {pipeline_id}...")
        query_record = generate_query_forward(
            difficulty=params["difficulty"],
            database_path=params["database_path"],
            query_style_file=params["query_style_file"],
            problem_family_file=params["problem_family_file"],
            guided_spec=guided_spec,
            llm_chat=llm_chat,
            model=params["model"],
            temperature=params["query_generation_forward_temperature"],
            seed=params["seed"],
            out_forward_jsonl=result_dir / "query" / "query_forward.jsonl",
        )
        
        if not query_record:
            raise ValueError("Failed to generate query")
        
        # Handle list return from generate_query_forward
        if isinstance(query_record, list):
            query_record = query_record[0]
        
        query_id = query_record["id"]

        # Step 3: Generate groundtruth (forward)
        if verbose:
            print(f"   📊 Generating groundtruth for query {query_id}...")
        groundtruth_record = generate_groundtruth_forward(
            query_id=query_id,
            query_bank_file=result_dir / "query" / "query_forward.jsonl",
            database_path=params["database_path"],
            llm_chat=llm_chat,
            model=params["model"],
            temperature=params["groundtruth_generation_temperature"],
            seed=params["seed"],
            out_forward_jsonl=result_dir / "groundtruth" / "groundtruth_forward.jsonl",
        )

        # Step 4: Verify format
        if verbose:
            print(f"   ✅ Verifying format for query {query_id}...")
        verify_format(
            query_id=query_id,
            query_bank_file=result_dir / "query" / "query_forward.jsonl",
            groundtruth_bank_file=result_dir / "groundtruth" / "groundtruth_forward.jsonl",
            database_path=params["database_path"],
            result_db_path=result_dir / "verdict",
        )

        # Step 5: Verify groundtruth (forward)
        if verbose:
            print(f"   🔍 Verifying groundtruth for query {query_id}...")
        verdict_forward = verify_groundtruth_forward(
            query_id=query_id,
            query_bank_path=result_dir / "query" / "query_forward.jsonl",
            result_db_path=result_dir / "verdict",
            groundtruth_bank_path=result_dir / "groundtruth" / "groundtruth_forward.jsonl",
            problem_family_file=params["problem_family_file"],
            llm_chat=llm_chat,
            model=params["model"],
            temperature=params["groundtruth_verification_temperature"],
            seed=params["seed"],
            out_jsonl=result_dir / "verdict" / "verdict_forward.jsonl",
        )
        
        # Step 6: Generate unit test (backward)
        if verbose:
            print(f"   🧪 Generating backward unit test for query {query_id}...")
        generate_groundtruth_backward(
            query_id=query_id,
            groundtruth_bank_file=result_dir / "groundtruth" / "groundtruth_forward.jsonl",
            verdict_path=result_dir / "verdict",
            query_bank_file=result_dir / "query" / "query_forward.jsonl",
            verdict_forward_file=result_dir / "verdict" / "verdict_forward.jsonl",
            llm_chat=llm_chat,
            model=params["model"],
            temperature=params["groundtruth_generation_temperature"],
            seed=params["seed"],
            out_backward_jsonl=result_dir / "groundtruth" / "groundtruth_backward.jsonl",
        )

        # Step 7: Generate query based on groundtruth (backward)
        if verbose:
            print(f"   🔄 Generating backward query for query {query_id}...")
        generate_query_backward(
            query_id=query_id,
            groundtruth_bank_file=result_dir / "groundtruth" / "groundtruth_backward.jsonl",
            verdict_path=result_dir / "verdict",
            database_path=params["database_path"],
            verdict_forward_file=result_dir / "verdict" / "verdict_forward.jsonl",
            llm_chat=llm_chat,
            model=params["model"],
            temperature=params["query_generation_backward_temperature"],
            seed=params["seed"],
            out_backward_jsonl=result_dir / "query" / "query_backward.jsonl",
        )

        # Step 8: Verify again (backward)
        if verbose:
            print(f"   🔍 Final backward verification for query {query_id}...")
        verdict_backward = verify_groundtruth_backward(
            query_id=query_id,
            result_db_path=result_dir / "verdict",
            groundtruth_bank_path=result_dir / "groundtruth" / "groundtruth_backward.jsonl",
            problem_family_file=params["problem_family_file"],
            out_jsonl=result_dir / "verdict" / "verdict_backward.jsonl",
        )
        
        # Determine success
        final_verdict = verdict_backward.get("verdict", "partial")
        final_adherence = verdict_backward.get("adherence", "partial")
        success = final_verdict == "correct" and final_adherence in ["adheres", "partial"]
        
        execution_time = time.time() - start_time

        if verbose:
            status = "✅" if success else "❌"
            print(f"   {status} Pipeline {pipeline_id} completed in {execution_time:.1f}s (success: {success})")

        return {
            "pipeline_id": pipeline_id,
            "query_id": query_id,
            "success": success,
            "verdict": final_verdict,
            "adherence": final_adherence,
            "execution_time": execution_time,
            "result_dir": str(result_dir),
        }

    except Exception as e:
        execution_time = time.time() - start_time
        import traceback
        error_details = f"{type(e).__name__}: {str(e)}"
        if not str(e):  # If error message is empty
            error_details = f"{type(e).__name__}: {repr(e)}"

        if verbose:
            print(f"   ❌ Pipeline {pipeline_id} failed after {execution_time:.1f}s: {error_details}")

        return {
            "pipeline_id": pipeline_id,
            "query_id": None,
            "success": False,
            "error": error_details,
            "execution_time": execution_time,
        }


def run_pipeline(
    difficulty: str,
    database_path: Union[str, Path],
    query_style_file: Union[str, Path],
    problem_family_file: Union[str, Path],
    result_base_path: Union[str, Path],
    batch_size: int,
    num_iteration: int,
    diversity_sampling: bool,
    sampling_tolerance: float,
    drop_incomplete: bool,
    model: str,
    query_generation_forward_temperature: float,
    groundtruth_generation_temperature: float,
    groundtruth_verification_temperature: float,
    query_generation_backward_temperature: float,
    seed: Optional[int],
    verbose: bool,
    single_problem_family: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute pipeline with multiple iterations and batch processing."""
    start_time = time.time()
    
    if verbose:
        print(f"🚀 Starting pipeline: {num_iteration} iteration(s) × {batch_size} batch size")
        if single_problem_family:
            print(f"🎯 Focusing on single problem family: {single_problem_family}")
        elif diversity_sampling:
            print(f"📊 Using diversity sampling (tolerance: ±{sampling_tolerance*100:.0f}%)")
    
    # Step 1: Setup sampler and create specification subsets
    total_pipelines = batch_size * num_iteration
    specification_subsets = []
    
    if (diversity_sampling or single_problem_family) and total_pipelines > 1:
        try:
            specification_subsets = create_specification_subsets(
                difficulty=difficulty,
                batch_size=batch_size,
                num_iteration=num_iteration,
                tolerance=sampling_tolerance,
                problem_family_file=problem_family_file,
                seed=seed,
                single_problem_family=single_problem_family
            )
            
            # Validate that all specification subsets are properly formed
            for i, spec in enumerate(specification_subsets):
                if spec is None:
                    raise ValueError(f"Specification subset {i} is None")
                if not isinstance(spec, dict):
                    raise ValueError(f"Specification subset {i} is not a dictionary: {type(spec)}")
                if len(spec) != 1:
                    raise ValueError(f"Specification subset {i} must have exactly one problem family: {list(spec.keys())}")
                
                # Validate the problem family structure
                problem_family_name = list(spec.keys())[0]
                family_features = spec[problem_family_name]
                if not isinstance(family_features, dict):
                    raise ValueError(f"Specification subset {i}: problem family '{problem_family_name}' must have dict of features")
            
            if verbose:
                print(f"✅ Created {len(specification_subsets)} specification subsets")
                
        except Exception as e:
            if verbose:
                print(f"❌ Failed to create specification subsets: {e}")
            raise RuntimeError(f"Sampler failed to create valid specification subsets: {e}")
    else:
        # For single pipeline or no diversity sampling, create a minimal valid spec
        from utils.utils import load_problem_family
        problem_family_config = load_problem_family(problem_family_file)
        problem_family_names = list(problem_family_config.get("problem_family_base", {}).keys())
        if not problem_family_names:
            raise ValueError("No problem families found in problem_family.json")
        
        # Use single_problem_family if specified, otherwise use first family as default
        default_family = single_problem_family if single_problem_family else problem_family_names[0]
        if single_problem_family and single_problem_family not in problem_family_names:
            raise ValueError(f"Single problem family '{single_problem_family}' not found. Available families: {problem_family_names}")
        
        specification_subsets = [{default_family: {}}] * total_pipelines
    
    # Setup result directory
    result_base = Path(result_base_path)
    result_base.mkdir(parents=True, exist_ok=True)
    for subdir in ["query", "groundtruth", "verdict"]:
        (result_base / subdir).mkdir(parents=True, exist_ok=True)
    
    # Tracking variables
    all_results = []
    total_successful = 0
    total_failed = 0
    
    # Parameters for pipeline functions
    params = {
        "difficulty": difficulty,
        "database_path": database_path,
        "query_style_file": query_style_file,
        "problem_family_file": problem_family_file,
        "model": model,
        "query_generation_forward_temperature": query_generation_forward_temperature,
        "groundtruth_generation_temperature": groundtruth_generation_temperature,
        "groundtruth_verification_temperature": groundtruth_verification_temperature,
        "query_generation_backward_temperature": query_generation_backward_temperature,
        "seed": seed,
    }
    
    # Run iterations
    for iteration in range(num_iteration):
        if verbose:
            print(f"\n🔄 Iteration {iteration + 1}/{num_iteration}")
        
        # Get specification subsets for this iteration
        start_idx = iteration * batch_size
        end_idx = start_idx + batch_size
        iteration_specs = specification_subsets[start_idx:end_idx]
        
        # Run batch pipelines for this iteration
        batch_results = []
        
        if batch_size == 1:
            # Serial execution
            pipeline_id = f"iter{iteration}_batch0"
            result = run_single_pipeline(
                pipeline_id=pipeline_id,
                guided_spec=iteration_specs[0],
                params=params,
                result_base_path=result_base,
                verbose=verbose
            )
            batch_results.append(result)
        else:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
                futures = []
                for batch_idx in range(batch_size):
                    pipeline_id = f"iter{iteration}_batch{batch_idx}"
                    guided_spec = iteration_specs[batch_idx] if batch_idx < len(iteration_specs) else None
                    
                    future = executor.submit(
                        run_single_pipeline,
                        pipeline_id=pipeline_id,
                        guided_spec=guided_spec,
                        params=params,
                        result_base_path=result_base,
                        verbose=verbose
                    )
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    batch_results.append(result)
        
        # Aggregate results
        iteration_successful = 0
        iteration_errors = []
        for result in batch_results:
            # Track success/failure for statistics
            if result.get("success", False):
                iteration_successful += 1
                total_successful += 1
            else:
                total_failed += 1
                # Collect error for debugging
                error_info = result.get("error", "Unknown error")
                pipeline_id = result.get("pipeline_id", "unknown")
                iteration_errors.append(f"{pipeline_id}: {error_info}")
        
        all_results.extend(batch_results)
        
        if verbose:
            print(f"   ✅ Completed: {iteration_successful}/{batch_size} successful")
            # Show errors for failed pipelines
            if iteration_errors:
                print(f"   ❌ Errors in failed pipelines:")
                for error in iteration_errors[:3]:  # Show first 3 errors
                    print(f"      • {error}")
                if len(iteration_errors) > 3:
                    print(f"      • ... and {len(iteration_errors) - 3} more errors")
    
    # Step 9: Save to dataset.jsonl if correct and adhere
    dataset_result = None
    if total_successful > 0:
        try:
            dataset_result = convert(
                result_base_path=result_base_path,
                out_jsonl=result_base / "dataset.jsonl",
                drop_incomplete=drop_incomplete,
            )
            if verbose:
                print(f"\n📄 Dataset conversion:")
                print(f"   📊 Records written: {dataset_result['written']}")
                print(f"   ⏭️  Records skipped: {dataset_result['skipped']}")
        except Exception as e:
            if verbose:
                print(f"\n⚠️  Dataset conversion failed: {e}")
            dataset_result = {"error": str(e)}
    
    execution_time = time.time() - start_time
    
    # Summary
    if verbose:
        print(f"\n🎯 Pipeline Complete!")
        print(f"   ⏱️  Total time: {execution_time:.1f}s")
        print(f"   📊 Total pipelines: {total_pipelines}")
        print(f"   ✅ Successful: {total_successful}")
        print(f"   ❌ Failed: {total_failed}")
        success_rate = total_successful / total_pipelines if total_pipelines > 0 else 0
        print(f"   🎯 Success rate: {success_rate:.1%}")
    
    return {
        "num_iteration": num_iteration,
        "batch_size": batch_size,
        "total_pipelines": total_pipelines,
        "total_successful": total_successful,
        "total_failed": total_failed,
        "success_rate": total_successful / total_pipelines if total_pipelines > 0 else 0,
        "execution_time": execution_time,
        "dataset_result": dataset_result,
        "all_results": all_results if not verbose else None,  # Include detailed results only in non-verbose mode
    }


def main():
    """CLI interface for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="SQL generation pipeline with 9-step workflow"
    )

    # Configuration file option (required)
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")

    # Core pipeline arguments (optional CLI overrides)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Override difficulty from config")
    parser.add_argument("--batch_size", type=int, help="Number of parallel pipelines per iteration")
    parser.add_argument("--num_iteration", type=int, help="Number of iterations to run")
    parser.add_argument("--no_diversity", action="store_true", help="Disable diversity sampling")

    args = parser.parse_args()

    # Load configuration from file (required)
    try:
        config = load_config(args.config)
        params = extract_config_params(config)
        
        # Get verbose setting from config for early logging
        verbose = params.get("verbose", True)
        if verbose:
            print(f"📋 Loading configuration from: {args.config}")

        # Apply CLI overrides only when explicitly provided
        cli_overrides = {}
        if args.difficulty is not None:
            cli_overrides["difficulty"] = args.difficulty    
        if args.batch_size is not None:
            cli_overrides["batch_size"] = args.batch_size
        if args.num_iteration is not None:
            cli_overrides["num_iteration"] = args.num_iteration
        if args.no_diversity:
            cli_overrides["diversity_sampling"] = False
        
        # Apply overrides
        params.update(cli_overrides)
        
        # Validate required parameters
        required_params = ["difficulty", "batch_size", "num_iteration"]
        missing_params = [p for p in required_params if not params.get(p)]
        if missing_params:
            print(f"❌ Error: Missing required parameters in config: {missing_params}")
            print("   Please ensure these are defined in the config file or provided via CLI")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error loading config file '{args.config}': {e}")
        sys.exit(1)

    # Run pipeline
    result = run_pipeline(**params)

    # Print appropriate output based on verbose setting
    if params.get("verbose", False):
        # Already printed during execution - just show completion
        pass
    else:
        # Full JSON output for programmatic use
        print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
