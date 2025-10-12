#!/usr/bin/env python3
"""
Sampler for ensuring diversity in SQL generation.

This module provides histogram tracking and intelligent sampling to distribute
problem family patterns across batches with uniform distribution.

Components:
- GenerationHistogram: Track problem family usage
- BinAwareSampler: Uniformly distribute problem families and select features based on difficulty
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter

from utils.utils import load_problem_family, get_difficulty_range


class GenerationHistogram:
    """
    Tracks counts of problem family usage across generations.
    Maintains histogram for the 7 problem families with uniform distribution.
    """
    
    def __init__(self, problem_family: Dict[str, Any]):
        """
        Initialize histogram with problem family configuration.
        Args:
            problem_family: Loaded problem family configuration
        """
        self.problem_family = problem_family
        
        # Extract problem family names from problem_family_base section
        self.problem_family_names = list(problem_family.get("problem_family_base", {}).keys())
        
        # Initialize problem family counters
        self.problem_family_bins = Counter()
        for name in self.problem_family_names:
            self.problem_family_bins[name] = 0
        
        self.total_samples = 0
    
    def update(self, problem_family_used: str) -> None:
        """
        Update histogram with newly generated sample.
        
        Args:
            problem_family_used: The problem family used for this sample
        """
        self.total_samples += 1
        
        if problem_family_used in self.problem_family_bins:
            self.problem_family_bins[problem_family_used] += 1
    
    def get_current_proportions(self) -> Dict[str, float]:
        """
        Get current proportions of problem family usage.
        
        Returns:
            Dict mapping problem family names to their current proportions
        """
        if self.total_samples == 0:
            return {name: 0.0 for name in self.problem_family_names}
        
        return {
            name: count / self.total_samples 
            for name, count in self.problem_family_bins.items()
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary of current histogram state.
        
        Returns:
            Dict with current counts, proportions, and bin information
        """
        proportions = self.get_current_proportions()
        
        return {
            "total_samples": self.total_samples,
            "problem_families": {
                "bins": len(self.problem_family_names),
                "counts": dict(self.problem_family_bins),
                "proportions": proportions,
            }
        }


class BinAwareSampler:
    """
    Intelligent sampler that uniformly distributes problem families and selects features based on difficulty.
    
    Creates problem family subsets for balanced generation with uniform distribution across 7 families.
    """
    
    def __init__(
        self, 
        problem_family_file: Union[str, Path],
        difficulty: str,
        batch_size: int,
        num_iteration: int,
        tolerance: float = 0.5,
        seed: Optional[int] = None,
        single_problem_family: Optional[str] = None
    ):
        """
        Initialize sampler with generation parameters.
        
        Args:
            problem_family_file: Path to problem family configuration file
            difficulty: Target difficulty level
            batch_size: Number of parallel pipeline runs per iteration
            num_iteration: Number of iterations (batches) total
            tolerance: Tolerance for uniform distribution variance (0.1 = ±10% variance allowed)
            seed: Random seed for reproducible sampling
            single_problem_family: If specified, focus only on this problem family instead of uniform distribution
        """
        self.problem_family = load_problem_family(problem_family_file)
        self.difficulty = difficulty
        self.batch_size = batch_size
        self.num_iteration = num_iteration
        self.tolerance = tolerance
        self.single_problem_family = single_problem_family
        self.rng = random.Random(seed)
        
        # Calculate total expected samples
        self.total_samples = batch_size * num_iteration
        
        # Get difficulty constraints
        self.difficulty_range = get_difficulty_range(problem_family_file, difficulty)
        self.min_difficulty_score = self.difficulty_range["min"]
        self.max_difficulty_score = self.difficulty_range["max"]
        
        # Extract problem family names and their base scores
        self.problem_family_names = list(self.problem_family.get("problem_family_base", {}).keys())
        self.problem_family_base_scores = self.problem_family.get("problem_family_base", {})
        
        # Validate single_problem_family if specified
        if self.single_problem_family:
            if self.single_problem_family not in self.problem_family_names:
                raise ValueError(f"Single problem family '{self.single_problem_family}' not found. Available families: {self.problem_family_names}")
        
        # Create distribution plan (uniform or single family)
        self.family_distribution = self._create_distribution()
    
    def _create_distribution(self) -> List[str]:
        """
        Create distribution of problem families across all samples.
        
        If single_problem_family is specified, all samples use that family.
        Otherwise, creates uniform distribution across all families.
        
        Returns:
            List of problem family names, one for each sample
        """
        if self.single_problem_family:
            # All samples use the single specified family
            return [self.single_problem_family] * self.total_samples
        else:
            # Use uniform distribution across all families
            return self._create_uniform_distribution()
    
    def _create_uniform_distribution(self) -> List[str]:
        """
        Create a uniform distribution of problem families across all samples.
        Uses tolerance to control acceptable variance in bin sizes.
        
        Returns:
            List of problem family names, one for each sample, distributed uniformly
        """
        if not self.problem_family_names:
            raise ValueError("No problem families found in configuration")
        
        num_families = len(self.problem_family_names)
        target_per_family = self.total_samples / num_families
        
        # Calculate base allocation and remainder
        base_samples_per_family = self.total_samples // num_families
        remainder = self.total_samples % num_families
        
        # Apply tolerance (as percentage) to determine acceptable range for each family
        # tolerance of 0.1 means ±10% variance from uniform distribution is allowed
        tolerance_range = int(target_per_family * self.tolerance)
        min_samples = max(0, base_samples_per_family - tolerance_range)
        max_samples = base_samples_per_family + tolerance_range + 1  # +1 for remainder handling
        
        # Distribute samples with tolerance consideration
        distribution = []
        families_to_allocate = self.problem_family_names.copy()
        self.rng.shuffle(families_to_allocate)  # Randomize family order for fair remainder distribution
        
        remaining_samples = self.total_samples
        
        for i, family in enumerate(families_to_allocate):
            remaining_families = len(families_to_allocate) - i
            
            if remaining_families == 1:
                # Last family gets all remaining samples
                count = remaining_samples
            else:
                # Calculate ideal allocation for this family
                ideal_count = remaining_samples / remaining_families
                
                # Apply tolerance constraints
                if remainder > 0 and ideal_count >= base_samples_per_family + 0.5:
                    # Give extra sample if we have remainder and it's close to ideal
                    count = min(max_samples, base_samples_per_family + 1)
                    remainder -= 1
                else:
                    count = max(min_samples, min(max_samples, base_samples_per_family))
            
            distribution.extend([family] * count)
            remaining_samples -= count
        
        # Shuffle to randomize order while maintaining allocation counts
        self.rng.shuffle(distribution)
        
        return distribution
    
    def _select_features_for_family(self, family_name: str) -> List[str]:
        """
        Select features for a given problem family that satisfy difficulty constraints.
        
        Args:
            family_name: Name of the problem family
            
        Returns:
            List of selected feature names
        """
        family_base_score = self.problem_family_base_scores.get(family_name, 0.0)
        available_budget = self.max_difficulty_score - family_base_score
        required_minimum = self.min_difficulty_score - family_base_score
        
        # Get all features for this family
        family_features = self.problem_family.get("family_features", {}).get(family_name, {})
        
        if not family_features:
            # If no features available, just use base family
            return []
        
        # Convert features to list with scores
        feature_options = []
        for feature_name, feature_config in family_features.items():
            feature_score = feature_config.get("difficulty_score", 0.0)
            if feature_score <= available_budget:  # Can fit within budget
                feature_options.append((feature_name, feature_score))
        
        if not feature_options:
            # No features fit within budget
            return []
        
        # Sort by score for greedy selection
        feature_options.sort(key=lambda x: x[1])
        
        selected_features = []
        current_score = 0.0
        
        # Greedy selection to meet minimum requirement
        for feature_name, feature_score in feature_options:
            if current_score + feature_score <= available_budget:
                selected_features.append(feature_name)
                current_score += feature_score
                
                # Stop if we've met the minimum and adding more would exceed max
                if current_score >= required_minimum:
                    # Try to add one more if it fits
                    remaining_options = [(fn, fs) for fn, fs in feature_options 
                                       if fn not in selected_features and current_score + fs <= available_budget]
                    if remaining_options and self.rng.random() < 0.3:  # 30% chance to add more
                        extra_feature, extra_score = self.rng.choice(remaining_options)
                        if current_score + extra_score <= available_budget:
                            selected_features.append(extra_feature)
                            current_score += extra_score
                    break
        
        return selected_features
    
    def create_problem_family_subsets(self) -> List[Dict[str, Any]]:
        """
        Create problem family subsets for each pipeline.
        
        Each subset contains one problem family with selected features that meet difficulty constraints.
        
        Returns:
            List of dictionaries in format: {problem_family_name: {feature_name: feature_config, ...}}
            Example: [
                {"relational_joining": {"join_count_2": {...}, "self_join": {...}}},
                {"aggregation_and_having": {"group_keys_2plus": {...}, "having_predicates": {...}}}
            ]
        """
        subsets = []
        
        # Create one subset for each sample based on uniform distribution
        for sample_idx in range(self.total_samples):
            # Get the assigned problem family for this sample
            assigned_family = self.family_distribution[sample_idx]
            
            # Select features for this family that meet difficulty constraints
            selected_features = self._select_features_for_family(assigned_family)
            
            # Calculate total difficulty
            family_base_score = self.problem_family_base_scores.get(assigned_family, 0.0)
            feature_scores = []
            for feature_name in selected_features:
                family_features = self.problem_family.get("family_features", {}).get(assigned_family, {})
                feature_config = family_features.get(feature_name, {})
                feature_scores.append(feature_config.get("difficulty_score", 0.0))
            
            total_difficulty = family_base_score + sum(feature_scores)
            
            # Create subset specification in the format: {problem_family: {selected_features}}
            selected_family_features = {}
            family_features_config = self.problem_family.get("family_features", {}).get(assigned_family, {})
            
            for feature_name in selected_features:
                if feature_name in family_features_config:
                    selected_family_features[feature_name] = family_features_config[feature_name]
            
            subset = {
                assigned_family: selected_family_features
            }
            
            subsets.append(subset)
        
        return subsets
    
    def get_sampling_summary(self) -> Dict[str, Any]:
        """
        Get summary of sampling configuration and targets.
        
        Returns:
            Dict with sampling configuration and target information
        """
        # Calculate distribution statistics
        family_counts = Counter(self.family_distribution)
        
        # Calculate tolerance-based distribution statistics
        num_families = len(self.problem_family_names)
        target_per_family = self.total_samples / num_families
        tolerance_range = int(target_per_family * self.tolerance)
        base_samples = self.total_samples // num_families
        
        return {
            "configuration": {
                "difficulty": self.difficulty,
                "batch_size": self.batch_size,
                "num_iteration": self.num_iteration,
                "total_samples": self.total_samples,
                "tolerance": self.tolerance,
                "min_difficulty_score": self.min_difficulty_score,
                "max_difficulty_score": self.max_difficulty_score
            },
            "problem_families": {
                "count": len(self.problem_family_names),
                "names": self.problem_family_names,
                "base_scores": self.problem_family_base_scores,
                "actual_distribution": dict(family_counts)
            },
            "tolerance_distribution": {
                "target_per_family": target_per_family,
                "base_samples_per_family": base_samples,
                "tolerance_range": tolerance_range,
                "min_allowed": max(0, base_samples - tolerance_range),
                "max_allowed": base_samples + tolerance_range + 1,
                "remainder": self.total_samples % num_families,
                "distribution_variance": {
                    family: abs(count - target_per_family) / target_per_family 
                    for family, count in family_counts.items()
                }
            }
        }


def create_sampler_from_config(
    problem_family_file: Union[str, Path],
    difficulty: str,
    batch_size: int,
    num_iteration: int,
    tolerance: float = 0.5,
    seed: Optional[int] = None,
    single_problem_family: Optional[str] = None
) -> Tuple[GenerationHistogram, BinAwareSampler]:
    """
    Create histogram and sampler from configuration.
    
    Args:
        problem_family_file: Path to problem_family.json
        difficulty: Target difficulty level
        batch_size: Number of parallel pipeline runs per iteration
        num_iteration: Number of iterations (batches) total
        tolerance: Tolerance for uniform distribution variance (0.1 = ±10% variance allowed)
        seed: Random seed for reproducible sampling
        single_problem_family: If specified, focus only on this problem family
        
    Returns:
        Tuple of (histogram, sampler)
    """
    problem_family = load_problem_family(problem_family_file)
    histogram = GenerationHistogram(problem_family)
    sampler = BinAwareSampler(
        problem_family_file=problem_family_file,
        difficulty=difficulty,
        batch_size=batch_size,
        num_iteration=num_iteration,
        tolerance=tolerance,
        seed=seed,
        single_problem_family=single_problem_family
    )
    
    return histogram, sampler


def create_specification_subsets(
    difficulty: str,
    batch_size: int,
    num_iteration: int,
    tolerance: float,
    problem_family_file: Union[str, Path],
    seed: Optional[int],
    single_problem_family: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Create specification subsets for guided generation.
    
    Args:
        difficulty: Target difficulty level
        batch_size: Number of parallel pipeline runs per iteration
        num_iteration: Number of iterations (batches) total
        tolerance: Tolerance for uniform distribution variance (0.1 = ±10% variance allowed)
        problem_family_file: Path to problem_family.json
        seed: Random seed for reproducible sampling
        single_problem_family: If specified, focus only on this problem family
        
    Returns:
        List of specification subsets in format: {problem_family_name: {feature_name: feature_config, ...}}
    """
    histogram, sampler = create_sampler_from_config(
        problem_family_file=problem_family_file,
        difficulty=difficulty,
        batch_size=batch_size,
        num_iteration=num_iteration,
        tolerance=tolerance,
        seed=seed,
        single_problem_family=single_problem_family
    )
    
    specification_subsets = sampler.create_problem_family_subsets()
    return specification_subsets


if __name__ == "__main__":
    # Example usage and testing
    print("🚀 Testing sampler.py...")
    
    # Test configuration
    test_config = {
        "problem_family_file": "problem_family.json",
        "difficulty": "medium",
        "batch_size": 4,
        "num_iteration": 3,
        "tolerance": 0.5,
        "seed": 42
    }
    
    # Create histogram and sampler
    print(f"📊 Creating sampler with config: {test_config}")
    histogram, sampler = create_sampler_from_config(**test_config)
    
    # Get sampling summary
    summary = sampler.get_sampling_summary()
    print(f"\n✅ Sampler initialized!")
    print(f"   Total expected samples: {summary['configuration']['total_samples']}")
    print(f"   Problem families available: {summary['problem_families']['count']}")
    print(f"   Problem family names: {summary['problem_families']['names']}")
    print(f"   Samples per family: {summary['uniform_distribution']['samples_per_family']}")
    print(f"   Difficulty range: {summary['configuration']['min_difficulty_score']}-{summary['configuration']['max_difficulty_score']}")
    
    # Create problem family subsets
    print(f"\n🎯 Creating problem family subsets...")
    subsets = sampler.create_problem_family_subsets()
    
    print(f"✅ Created {len(subsets)} problem family subsets:")
    
    # Show sample subsets
    for i, subset in enumerate(subsets[:5]):  # Show first 5 for brevity
        # Extract family name and features from new format
        family_name = list(subset.keys())[0]
        family_features = subset[family_name]
        
        # Calculate base score for display
        base_score = sampler.problem_family_base_scores.get(family_name, 0.0)
        
        print(f"   Sample {i+1}: Family '{family_name}' (base: {base_score:.1f})")
        print(f"      Features: {list(family_features.keys())}")
        
        # Calculate total difficulty for display
        feature_total = sum(config.get("difficulty_score", 0.0) for config in family_features.values())
        total_difficulty = base_score + feature_total
        print(f"      Total difficulty: {total_difficulty:.1f}")
    
    if len(subsets) > 5:
        print(f"   ... and {len(subsets) - 5} more")
    
    # Show tolerance-based distribution analysis
    print(f"\n📊 Tolerance-based family distribution analysis:")
    family_counts = Counter()
    for subset in subsets:
        family_name = list(subset.keys())[0]
        family_counts[family_name] += 1
    
    summary = sampler.get_sampling_summary()
    tolerance_info = summary["tolerance_distribution"]
    
    print(f"   Target per family: {tolerance_info['target_per_family']:.1f}")
    print(f"   Tolerance range: ±{tolerance_info['tolerance_range']}")
    print(f"   Allowed range: {tolerance_info['min_allowed']}-{tolerance_info['max_allowed']}")
    
    print(f"\n   Actual distribution:")
    for family, count in family_counts.items():
        variance = tolerance_info['distribution_variance'][family]
        status = "✅" if variance <= test_config['tolerance'] else "⚠️"
        print(f"   {status} {family}: {count} samples (variance: {variance:.1%})")
    
    # Test histogram tracking
    print(f"\n📈 Testing histogram tracking...")
    
    # Simulate some sample updates with problem families
    test_families = [
        "relational_joining",
        "aggregation_and_having", 
        "set_algebra",
        "subquery_semantics"
    ]
    
    for family in test_families:
        histogram.update(family)
    
    status = histogram.get_status_summary()
    print(f"✅ Histogram tracking test complete!")
    print(f"   Total samples tracked: {status['total_samples']}")
    print(f"   Problem family bins used: {sum(1 for count in status['problem_families']['counts'].values() if count > 0)}")
    print(f"   Problem family counts: {status['problem_families']['counts']}")
    
    print(f"\n🎯 Sampler.py testing complete! Ready for production use.")