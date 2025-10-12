#!/usr/bin/env python3
"""
Training File Wrapper for Manufactoria Problems

Converts JSONL ensemble files into the required training format with configurable prompt templates.
"""

import json
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from datasets import Dataset
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TrainingFileWrapper:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the training file wrapper.
        
        Args:
            config: Configuration dictionary containing template settings
        """
        self.config = config or {}
        self.prompt_templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load different prompt templates based on configuration."""
        
        # Base DSL template (from dsl.md)
        base_dsl_template = """# Manufactoria Solution DSL

A Domain Specific Language for describing Manufactoria puzzle solutions in text format.

## Overview

Manufactoria is a puzzle game where you build automated factories to sort robots based on their colored tape patterns. Robots enter your factory carrying sequences of colored tape, and you must route them to the correct destinations based on the given criteria.

## Game Mechanics

### Robots and Tape
- **Robots**: Each robot carries a sequence of colored tapes
- **Tape Colors**: Primary colors are Blue (B) and Red (R), with additional Yellow (Y) and Green (G) for advanced puzzles
- **Tape Representation**: Sequences are represented as strings (e.g., `RBRR`, `BBR`, or empty string `""`)

### Operations
- **Pull**: Remove tape from the front of the robot's sequence
- **Paint**: Add colored tape to the end of the robot's sequence
- **Route**: Direct robots through the factory based on their current tape state

{objective_section} 

## DSL Syntax

### Program Structure

Every solution must start with a `START` directive and end with an `END` directive, wrapped in ```manufactoria ...```:

```manufactoria
START start:
    NEXT <next_node_id>

# Factory logic goes here

END end
```

### Node Types

#### 1. Puller Nodes

Pullers remove specific colors from the front of the robot's tape sequence and route based on the current front color.

**Red/Blue Puller:**

```manufactoria
PULLER_RB <node_id>:
    [R] <next_node_id>      # Route and remove color if front tape is Red
    [B] <next_node_id>      # Route and remove color if front tape is Blue
    [EMPTY] <next_node_id>  # Route if no tape or front tape is neither red nor blue
```

**Yellow/Green Puller:**

```manufactoria
PULLER_YG <node_id>:
    [Y] <next_node_id>      # Route and remove color if front tape is Yellow
    [G] <next_node_id>      # Route and remove color if front tape is Green
    [EMPTY] <next_node_id>  # Route if no tape or front tape is neither yellow nor green
```

**Note**: Unspecified branches default to `NONE`, which rejects the robot.

#### 2. Painter Nodes

Painters add colored tape to the end of the robot's sequence and continue to the next node.

```manufactoria
PAINTER_RED <node_id>:
    NEXT <next_node_id>

PAINTER_BLUE <node_id>:
    NEXT <next_node_id>

PAINTER_YELLOW <node_id>:
    NEXT <next_node_id>

PAINTER_GREEN <node_id>:
    NEXT <next_node_id>
```

## Syntax Rules

1. **Node IDs**: Must be unique identifiers (alphanumeric characters and underscores only)
2. **Comments**: Lines starting with `#` are comments (single-line only)
3. **Indentation**: Use consistent spaces or tabs for route definitions
4. **Case Sensitivity**: Colors must be uppercase (R, B, Y, G)
5. **Termination**: 
   - Robots routed to `NONE` are rejected
   - Robots routed to the END node are accepted{objective_clause}
6. **Code Blocks**: Final factory code should be wrapped in triple backticks with ``` markers

## Example

Here's a simple example that accepts robots with exactly one red tape (ending tape should be empty):

```manufactoria
START start:
    NEXT entry

PULLER_RB entry:
    [R] end

END end
```

# Task 
Your task is to design a factory with code with following functionality:

{criteria}"""

        # Different objective sections based on output checking
        objective_simple = """### Objective
Route robots to the correct destinations based on their final tape configuration and the puzzle requirements:
- **Accepted**: Robot reaches the END node
- **Rejected**: Robot is not routed to the END node or caught in an infinite loop."""

        objective_with_output_check = """### Objective (output check)
Route robots to the correct destinations based on their final tape configuration and the puzzle requirements:
- **Accepted**: Robot reaches the END node and meets the puzzle's acceptance criteria
- **Rejected**: Robot is routed to the NONE node, or caught in an infinite loop, or robot reaches the END node but fails to meet the puzzle's acceptance criteria"""

        # Different syntax rules clauses based on output checking  
        syntax_rules_simple = ""
        syntax_rules_with_output_check = " if they meet the puzzle criteria, otherwise rejected"
        
        return {
            "simple": base_dsl_template.replace("{objective_section}", objective_simple).replace("{objective_clause}", syntax_rules_simple),
            "output_check": base_dsl_template.replace("{objective_section}", objective_with_output_check).replace("{objective_clause}", syntax_rules_with_output_check)
        }
    
    def _get_prompt_template(self, has_output_check: bool) -> str:
        """Get the appropriate prompt template based on output checking requirement."""
        template_type = "output_check" if has_output_check else "simple"
        return self.prompt_templates[template_type]
    
    def _extract_test_cases(self, problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and format test cases from problem data."""
        test_cases = []
        for test_case in problem_data.get("test_cases", []):
            formatted_case = {
                "input": test_case.get("input", ""),
                "expected_output": test_case.get("expected_output", ""),
                "expected_accepted": test_case.get("expected_accepted", False),
                "check_output": test_case.get("check_output", False),
                "description": test_case.get("description", "")
            }
            test_cases.append(formatted_case)
        return test_cases
    
    def _determine_difficulty(self, problem_data: Dict[str, Any]) -> str:
        """Determine difficulty level from problem data."""
        # Try multiple fields for difficulty
        difficulty = problem_data.get("difficulty_level")
        return str(difficulty)
    
    def _determine_problem_family(self, problem_data: Dict[str, Any]) -> str:
        """Determine problem family from problem data."""
        # Try multiple fields for problem family
        family = (
            problem_data.get("problem_type") or
            problem_data.get("pattern_type") or
            problem_data.get("_ensemble_source", {}).get("family") or
            "unknown"
        )
        return str(family)
    
    def _has_output_check(self, test_cases: List[Dict[str, Any]]) -> bool:
        """Determine if any test case requires output checking."""
        return any(test_case.get("check_output", False) for test_case in test_cases)
    
    def convert_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single problem from JSONL format to training format.
        
        Args:
            problem_data: Single problem data from JSONL file
            
        Returns:
            Formatted training example
        """
        test_cases = self._extract_test_cases(problem_data)
        has_output_check = self._has_output_check(test_cases)
        
        # Get the appropriate prompt template
        prompt_template = self._get_prompt_template(has_output_check)
        
        # Format the prompt with the criteria
        criteria = problem_data.get("criteria", "")
        formatted_prompt = prompt_template.format(criteria=criteria)
        
        # Create the training example
        training_example = {
            "messages": [
                {
                    "content": formatted_prompt,
                    "role": "user"
                }
            ],
            "ground_truth": test_cases,
            "dataset": "manufactoria",
            "difficulty": self._determine_difficulty(problem_data),
            "id": problem_data.get("id", ""),
            "problem_family": self._determine_problem_family(problem_data),
            "name": problem_data.get("name", "")
        }
        
        return training_example
    
    def convert_jsonl_file(self, input_file: str, output_file: str) -> List[Dict[str, Any]]:
        """
        Convert an entire JSONL file to training format.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output training file
            
        Returns:
            List of training examples
        """
        training_examples = []
        
        print(f"Converting {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if line:
                    try:
                        problem_data = json.loads(line)
                        training_example = self.convert_problem(problem_data)
                        training_examples.append(training_example)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_count + 1}: {e}")
        
        # Add manufactoria_ prefix to output file
        output_path = Path(output_file)
        manufactoria_output = output_path.parent / f"manufactoria_{output_path.name}"
        
        # Write to output file
        with open(manufactoria_output, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2, ensure_ascii=False)
        
        print(f"Converted {line_count} problems to {manufactoria_output}")
        return training_examples
    
    def convert_ensemble_directory(self, ensemble_dir: str, output_dir: str) -> List[str]:
        """
        Convert all JSONL files in the ensemble directory.
        
        Args:
            ensemble_dir: Path to directory containing JSONL files
            output_dir: Path to output directory for training files
            
        Returns:
            List of output file paths
        """
        ensemble_path = Path(ensemble_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all JSONL files
        jsonl_files = list(ensemble_path.glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"No JSONL files found in {ensemble_dir}")
            return []
        
        print(f"Found {len(jsonl_files)} JSONL files to convert")
        
        output_files = []
        for jsonl_file in jsonl_files:
            # Generate output filename
            output_filename = f"{jsonl_file.stem}.json"
            output_file = output_path / output_filename
            
            try:
                self.convert_jsonl_file(str(jsonl_file), str(output_file))
                # The actual output file will have manufactoria_ prefix
                actual_output = output_path / f"manufactoria_{output_filename}"
                output_files.append(str(actual_output))
            except Exception as e:
                print(f"Error converting {jsonl_file}: {e}")
        
        print(f"Conversion complete. Training files saved to {output_dir}")
        return output_files
    
    def check_repository_exists(self, repo_id: str, token: Optional[str] = None, repo_type: str = "dataset") -> bool:
        """
        Check if a repository exists on Hugging Face Hub.
        
        Args:
            repo_id: Repository ID in format "namespace/name"
            token: HF token (will use environment variable if not provided)
            repo_type: Type of repository ("dataset", "model", "space")
            
        Returns:
            True if repository exists, False otherwise
        """
        if not HF_AVAILABLE:
            print("Error: huggingface_hub package is required for repository checking.")
            return False
        
        # Get token from environment if not provided
        if token is None:
            token = os.getenv("HF_TOKEN")
        
        try:
            # Create HfApi instance
            api = HfApi(token=token)
            
            # Try to get repository info - this will raise RepositoryNotFoundError if repo doesn't exist
            if repo_type == "dataset":
                api.dataset_info(repo_id)
            elif repo_type == "model":
                api.model_info(repo_id)
            elif repo_type == "space":
                api.space_info(repo_id)
            else:
                # Default to model if repo_type is not recognized
                api.model_info(repo_id)
            
            return True
            
        except RepositoryNotFoundError:
            # Repository doesn't exist
            return False
        except Exception as e:
            print(f"Error checking repository existence: {e}")
            return False
    
    def upload_to_huggingface(self, file_path: str, dataset_name: str, namespace: str = "sunyiyou", token: Optional[str] = None, private: bool = False, force_overwrite: bool = False) -> bool:
        """
        Upload a training file to Hugging Face Hub.
        
        Args:
            file_path: Path to the training file
            dataset_name: Name for the dataset on HF Hub
            namespace: HF namespace/organization name
            token: HF token (will use environment variable if not provided)
            private: Whether to upload as private dataset
            force_overwrite: Whether to overwrite existing repositories
            
        Returns:
            True if upload successful, False otherwise
        """
        if not HF_AVAILABLE:
            print("Error: huggingface_hub and datasets packages are required for upload.")
            print("Install with: pip install datasets huggingface_hub")
            return False
        
        # Get token from environment if not provided
        if token is None:
            token = os.getenv("HF_TOKEN")
            if not token:
                print("Error: No HF token provided. Set HF_TOKEN environment variable or pass token directly.")
                return False
        
        # Check if repository already exists
        repo_id = f"{namespace}/{dataset_name}"
        print(f"Checking if repository {repo_id} already exists...")
        
        if self.check_repository_exists(repo_id, token, "dataset"):
            if not force_overwrite:
                print(f"Repository {repo_id} already exists. Skipping upload to avoid rate limits.")
                print(f"Existing repository: https://huggingface.co/datasets/{repo_id}")
                print("Use --force-overwrite to overwrite existing repositories.")
                return True  # Return True since the dataset exists (successful state)
            else:
                print(f"Repository {repo_id} already exists. Force overwrite enabled, proceeding with upload...")
        
        try:
            print(f"Repository {repo_id} does not exist. Proceeding with upload...")
            print(f"Loading dataset from {file_path}...")
            
            # Load the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create dataset
            dataset = Dataset.from_list(data)
            
            # Upload to Hub
            print(f"Uploading to {repo_id}...")
            
            dataset.push_to_hub(
                repo_id=repo_id,
                token=token,
                # private=private
            )
            
            print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
            return True
            
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
            return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Convert Manufactoria JSONL files to training format"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="problems/random",
        help="Input JSONL file or directory containing JSONL files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="problems/wrapped",
        help="Output file or directory for training files"
    )
    
    parser.add_argument(
        "--config",
        help="Optional configuration file for custom templates"
    )
    
    parser.add_argument(
        "--template-type",
        choices=["simple", "output_check", "auto"],
        default="auto",
        help="Force a specific template type (auto detects based on test cases)"
    )
    
    parser.add_argument(
        "--upload",
        default=True,
        help="Upload the converted dataset to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--dataset-name",
        help="Name for the dataset on HF Hub (uses original file name if not provided)"
    )
    
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (will use HF_TOKEN environment variable if not provided)"
    )
    
    parser.add_argument(
        "--private",
        default=False,
        help="Upload as private dataset (default is public)"
    )
    
    parser.add_argument(
        "--hf-namespace",
        default="manufactoria",
        help="Hugging Face namespace/organization (default: manufactoria)"
    )
    
    parser.add_argument(
        "--force-overwrite",
        default=True,
        help="Force overwrite existing HuggingFace repositories"
    )
    
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=3,
        help="Number of seconds to sleep after uploading each repository (to handle rate limits)"
    )
    
    args = parser.parse_args()
    
    # No validation needed since dataset-name is now optional
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override template type if specified
    if args.template_type != "auto":
        config["force_template_type"] = args.template_type
    
    # Create wrapper
    wrapper = TrainingFileWrapper(config)
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    output_files = []

    if input_path.is_file():
        # Single file conversion
        wrapper.convert_jsonl_file(args.input, args.output)
        # Add manufactoria_ prefix to get actual output path
        output_path = Path(args.output)
        actual_output = output_path.parent / f"manufactoria_{output_path.name}"
        output_files = [str(actual_output)]
    elif input_path.is_dir():
        # Directory conversion
        output_files = wrapper.convert_ensemble_directory(args.input, args.output)
    else:
        print(f"Error: Input path {args.input} does not exist")
        return 1
    
    # Upload to Hugging Face if requested
    if args.upload:
        success_count = 0
        valid_files = [f for f in output_files if os.path.exists(f)]
        
        for i, output_file in enumerate(valid_files):
            # Generate dataset name
            file_stem = Path(output_file).stem.replace("manufactoria_", "")
            
            if args.dataset_name:
                # Use provided dataset name
                if len(output_files) == 1:
                    dataset_name = args.dataset_name
                else:
                    # For multiple files, append file stem to dataset name
                    dataset_name = f"{args.dataset_name}_{file_stem}"
            else:
                # Use original file name if no dataset name provided
                dataset_name = file_stem
            
            success = wrapper.upload_to_huggingface(
                output_file, 
                dataset_name, 
                args.hf_namespace,
                args.hf_token,
                args.private,
                args.force_overwrite
            )
            if success:
                success_count += 1
            
            # Sleep after upload to handle rate limits (but not after the last upload)
            if args.sleep_seconds > 0 and i < len(valid_files) - 1:
                print(f"Sleeping for {args.sleep_seconds} seconds to handle rate limits...")
                time.sleep(args.sleep_seconds)
        
        print(f"\nUploaded {success_count}/{len(output_files)} datasets to Hugging Face")
    
    return 0


if __name__ == "__main__":
    exit(main())
