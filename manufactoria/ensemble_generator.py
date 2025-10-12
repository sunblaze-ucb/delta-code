#!/usr/bin/env python3
"""
Ensemble Generator for Manufactoria Problems

This script creates mixed datasets from different problem families with configurable
difficulty levels and sample counts. It supports train/test splits and various 
combination strategies.

Usage:
    python ensemble_generator.py --config ensemble_config.yaml
    python ensemble_generator.py --interactive
"""

import json
import yaml
import random
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ProblemSpec:
    """Specification for a problem family in the ensemble."""
    family: str
    difficulty: str
    color_mode: Optional[str]
    sample_count: int
    
    def __post_init__(self):
        """Validate the specification."""
        if self.sample_count <= 0:
            raise ValueError(f"Sample count must be positive, got {self.sample_count}")

@dataclass
class EnsembleConfig:
    """Configuration for ensemble generation."""
    name: str
    problem_specs: List[ProblemSpec]
    train_ratio: float = 0.8
    output_dir: str = "ensembles"
    seed: Optional[int] = None
    shuffle: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.train_ratio <= 1:
            raise ValueError(f"Train ratio must be between 0 and 1 (inclusive), got {self.train_ratio}")
        if not self.problem_specs:
            raise ValueError("Must specify at least one problem specification")

class EnsembleGenerator:
    """Main class for generating ensemble datasets."""
    
    def __init__(self, difficulty_dir: str = "problems/difficulty"):
        self.difficulty_dir = Path(difficulty_dir)
        self.available_files = self._discover_available_files()
        
    def _discover_available_files(self) -> Dict[str, Dict[str, str]]:
        """Discover all available problem files and their metadata."""
        files = {}
        for file_path in self.difficulty_dir.glob("*.jsonl"):
            # Parse filename: family_colormode_difficulty.jsonl
            parts = file_path.stem.split("_")
            if len(parts) >= 3:
                # Handle cases like "numerical_operations_two_color_easy"
                if "color" in parts:
                    color_idx = next(i for i, part in enumerate(parts) if part == "color")
                    family = "_".join(parts[:color_idx-1])
                    color_mode = f"{parts[color_idx-1]}_color"
                    difficulty = "_".join(parts[color_idx+1:])
                else:
                    # Fallback for other patterns
                    family = "_".join(parts[:-1])
                    color_mode = None
                    difficulty = parts[-1]
                
                key = f"{family}_{color_mode}_{difficulty}" if color_mode else f"{family}_{difficulty}"
                files[key] = {
                    "path": str(file_path),
                    "family": family,
                    "difficulty": difficulty,
                    "color_mode": color_mode
                }
        return files
    
    def _count_problems_in_file(self, file_path: str) -> int:
        """Count the number of problems in a JSONL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except (FileNotFoundError, PermissionError):
            return 0
    
    def list_available_problems(self) -> None:
        """Print all available problem families and their configurations with problem counts."""
        families = defaultdict(lambda: defaultdict(list))
        file_counts = {}
        
        # Collect family information and count problems
        for key, file_info in self.available_files.items():
            family = file_info["family"]
            color_mode = file_info["color_mode"] or "default"
            difficulty = file_info["difficulty"]
            file_path = file_info["path"]
            
            # Count problems in this file
            count = self._count_problems_in_file(file_path)
            file_counts[key] = count
            
            families[family][color_mode].append((difficulty, count))
        
        print("Available Problem Families:")
        print("=" * 70)
        total_problems = 0
        
        for family, color_modes in sorted(families.items()):
            print(f"\n{family}:")
            for color_mode, difficulty_counts in sorted(color_modes.items()):
                # Sort by difficulty name, then format with counts
                sorted_diffs = sorted(difficulty_counts, key=lambda x: x[0])
                diff_strings = [f"{diff} ({count} problems)" for diff, count in sorted_diffs]
                print(f"  {color_mode}: {', '.join(diff_strings)}")
                total_problems += sum(count for _, count in sorted_diffs)
        
        print(f"\nTotal problems across all families: {total_problems}")
    
    def _find_file_for_spec(self, spec: ProblemSpec) -> Optional[str]:
        """Find the file path for a given problem specification."""
        # Try different key combinations
        possible_keys = []
        if spec.color_mode:
            possible_keys.append(f"{spec.family}_{spec.color_mode}_{spec.difficulty}")
        possible_keys.append(f"{spec.family}_{spec.difficulty}")
        
        for key in possible_keys:
            if key in self.available_files:
                return self.available_files[key]["path"]
        
        return None
    
    def _load_problems_from_file(self, file_path: str, sample_count: int) -> List[Dict[str, Any]]:
        """Load a specified number of problems from a JSONL file."""
        problems = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_problems = [json.loads(line.strip()) for line in f if line.strip()]
            
            if len(all_problems) < sample_count:
                print(f"Warning: File {file_path} has only {len(all_problems)} problems, "
                      f"but {sample_count} requested. Using all available.")
                return all_problems
            else:
                return random.sample(all_problems, sample_count)
                
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {file_path}: {e}")
            return []
    
    def generate_ensemble(self, config: EnsembleConfig) -> Tuple[List[Dict], List[Dict]]:
        """Generate an ensemble dataset according to the configuration."""
        if config.seed is not None:
            random.seed(config.seed)
        
        all_problems = []
        
        print(f"Generating ensemble '{config.name}'...")
        print("-" * 40)
        
        for spec in config.problem_specs:
            file_path = self._find_file_for_spec(spec)
            if not file_path:
                print(f"Error: Could not find file for {spec.family} "
                      f"(difficulty: {spec.difficulty}, color_mode: {spec.color_mode})")
                continue
            
            problems = self._load_problems_from_file(file_path, spec.sample_count)
            if problems:
                # Add metadata to track source
                for problem in problems:
                    problem["_ensemble_source"] = {
                        "family": spec.family,
                        "difficulty": spec.difficulty,
                        "color_mode": spec.color_mode,
                        "file_path": file_path
                    }
                all_problems.extend(problems)
                print(f"✓ {spec.family} ({spec.difficulty}): {len(problems)} problems")
            else:
                print(f"✗ {spec.family} ({spec.difficulty}): No problems loaded")
        
        if config.shuffle:
            random.shuffle(all_problems)
        
        # Split into train and test
        if config.train_ratio == 1.0:
            # Only training data
            train_set = all_problems
            test_set = []
        elif config.train_ratio == 0.0:
            # Only test data
            train_set = []
            test_set = all_problems
        else:
            # Normal split
            train_size = int(len(all_problems) * config.train_ratio)
            train_set = all_problems[:train_size]
            test_set = all_problems[train_size:]
        
        print(f"\nEnsemble generated:")
        print(f"  Total problems: {len(all_problems)}")
        if train_set:
            print(f"  Train set: {len(train_set)} problems")
        if test_set:
            print(f"  Test set: {len(test_set)} problems")
        
        return train_set, test_set
    
    def save_ensemble(self, train_set: List[Dict], test_set: List[Dict], config: EnsembleConfig) -> None:
        """Save the ensemble to files."""
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        saved_files = []
        metadata = {
            "name": config.name,
            "train_count": len(train_set),
            "test_count": len(test_set),
            "train_ratio": config.train_ratio,
            "problem_specs": [
                {
                    "family": spec.family,
                    "difficulty": spec.difficulty,
                    "color_mode": spec.color_mode,
                    "sample_count": spec.sample_count
                }
                for spec in config.problem_specs
            ],
            "seed": config.seed
        }
        
        # Save train set if it exists
        if train_set:
            train_path = output_dir / f"{config.name}_train.jsonl"
            with open(train_path, 'w', encoding='utf-8') as f:
                for problem in train_set:
                    f.write(json.dumps(problem) + '\n')
            metadata["train_file"] = str(train_path)
            saved_files.append(f"Train: {train_path}")
        
        # Save test set if it exists
        if test_set:
            test_path = output_dir / f"{config.name}_test.jsonl"
            with open(test_path, 'w', encoding='utf-8') as f:
                for problem in test_set:
                    f.write(json.dumps(problem) + '\n')
            metadata["test_file"] = str(test_path)
            saved_files.append(f"Test: {test_path}")
        
        # Save metadata
        metadata_path = output_dir / f"{config.name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        saved_files.append(f"Metadata: {metadata_path}")
        
        print(f"\nEnsemble saved:")
        for file_info in saved_files:
            print(f"  {file_info}")

def load_config_from_yaml(config_path: str) -> EnsembleConfig:
    """Load ensemble configuration from a YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    problem_specs = []
    for spec_data in data.get("problem_specs", []):
        problem_specs.append(ProblemSpec(
            family=spec_data["family"],
            difficulty=spec_data["difficulty"],
            color_mode=spec_data.get("color_mode"),
            sample_count=spec_data["sample_count"]
        ))
    
    return EnsembleConfig(
        name=data["name"],
        problem_specs=problem_specs,
        train_ratio=data.get("train_ratio", 0.8),
        output_dir=data.get("output_dir", "ensembles"),
        seed=data.get("seed"),
        shuffle=data.get("shuffle", True)
    )

def interactive_config() -> EnsembleConfig:
    """Create configuration interactively."""
    generator = EnsembleGenerator()
    
    print("Interactive Ensemble Configuration")
    print("=" * 40)
    
    # Show available problems
    generator.list_available_problems()
    
    name = input("\nEnsemble name: ").strip()
    
    problem_specs = []
    print("\nAdd problem specifications (press Enter with empty family to finish):")
    
    while True:
        family = input("  Problem family: ").strip()
        if not family:
            break
        
        difficulty = input("  Difficulty level: ").strip()
        color_mode = input("  Color mode (optional, press Enter to skip): ").strip() or None
        
        while True:
            try:
                sample_count = int(input("  Sample count: ").strip())
                break
            except ValueError:
                print("    Please enter a valid number.")
        
        problem_specs.append(ProblemSpec(family, difficulty, color_mode, sample_count))
        print("  ✓ Added\n")
    
    if not problem_specs:
        raise ValueError("Must specify at least one problem specification")
    
    # Optional parameters
    train_ratio = 0.8
    try:
        ratio_input = input(f"Train ratio (default {train_ratio}): ").strip()
        if ratio_input:
            train_ratio = float(ratio_input)
    except ValueError:
        print(f"Invalid ratio, using default {train_ratio}")
    
    output_dir = input("Output directory (default 'ensembles'): ").strip() or "ensembles"
    
    seed_input = input("Random seed (optional): ").strip()
    seed = int(seed_input) if seed_input else None
    
    return EnsembleConfig(
        name=name,
        problem_specs=problem_specs,
        train_ratio=train_ratio,
        output_dir=output_dir,
        seed=seed
    )

def main():
    parser = argparse.ArgumentParser(description="Generate ensemble datasets from Manufactoria problems")
    parser.add_argument("--config", "-c", help="Path to configuration YAML file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive configuration mode")
    parser.add_argument("--list", "-l", action="store_true", help="List available problem families")
    parser.add_argument("--difficulty-dir", "-d", default="problems/difficulty", 
                       help="Directory containing difficulty-organized problems")
    
    args = parser.parse_args()
    
    generator = EnsembleGenerator(args.difficulty_dir)
    
    if args.list:
        generator.list_available_problems()
        return
    
    if args.interactive:
        config = interactive_config()
    elif args.config:
        config = load_config_from_yaml(args.config)
    else:
        parser.print_help()
        return
    
    try:
        train_set, test_set = generator.generate_ensemble(config)
        generator.save_ensemble(train_set, test_set, config)
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())