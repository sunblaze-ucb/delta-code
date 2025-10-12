#!/usr/bin/env python3
"""
Test visualization for sampler.py histogram distribution.

This script demonstrates how the BinAwareSampler distributes bins across batches
and visualizes the expected histogram distribution for sanity checks.

Usage:
    python analyze/test_sampler_visualization.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import sampler
sys.path.append(str(Path(__file__).parent.parent))
from sampler import create_sampler_from_config


def visualize_bin_distribution(histogram, sampler, subsets):
    """
    Visualize key metrics for sampler distribution (simplified output).
    
    Args:
        histogram: GenerationHistogram instance
        sampler: BinAwareSampler instance  
        subsets: List of problem family subsets from sampler
    """
    summary = sampler.get_sampling_summary()
    config = summary['configuration']
    
    print(f"🎯 {config['difficulty'].upper()} | {config['batch_size']}×{sampler.num_iteration} = {config['total_samples']} samples")
    print(f"   Available: {summary['available_options']['complexity']} complexity + {summary['available_options']['task_types']} task_type bins")
    
    # Track batch coverage
    complexity_batch_map = defaultdict(list)
    task_type_batch_map = defaultdict(list)
    
    for i, subset in enumerate(subsets):
        meta = subset['batch_metadata']
        for comp in meta['selected_complexity']:
            complexity_batch_map[comp].append(i+1)
        for task in meta['selected_task_types']:
            task_type_batch_map[task].append(i+1)
    
    # Simulate distribution
    samples_per_batch = config['total_samples'] // len(subsets)
    remainder = config['total_samples'] % len(subsets)
    
    simulated_complexity = defaultdict(int)
    for i, subset in enumerate(subsets):
        meta = subset['batch_metadata']
        batch_samples = samples_per_batch + (1 if i < remainder else 0)
        
        if meta['selected_complexity']:
            samples_per_complexity = batch_samples // len(meta['selected_complexity'])
            for comp in meta['selected_complexity']:
                simulated_complexity[comp] += samples_per_complexity
    
    # Analyze tolerance
    complexity_targets = summary['targets']['complexity']
    total_simulated = sum(simulated_complexity.values())
    complexity_within_tolerance = 0
    total_complexity_bins = 0
    
    for name, target_count in complexity_targets.items():
        if target_count > 0:
            total_complexity_bins += 1
            simulated_count = simulated_complexity.get(name, 0)
            actual_prop = simulated_count / total_simulated if total_simulated > 0 else 0
            target_prop = target_count / config['total_samples'] if config['total_samples'] > 0 else 0
            
            deviation = abs(actual_prop - target_prop)
            if target_prop > 0 and deviation <= target_prop * config['tolerance']:
                complexity_within_tolerance += 1
    
    # Coverage analysis
    high_coverage_complexity = sum(1 for batches in complexity_batch_map.values() 
                                 if len(batches) / len(subsets) >= 0.8)
    
    success_rate = complexity_within_tolerance / total_complexity_bins * 100 if total_complexity_bins > 0 else 0
    coverage_rate = high_coverage_complexity / len(complexity_batch_map) * 100 if complexity_batch_map else 0
    
    # Status indicator
    if success_rate >= 70 and coverage_rate >= 60:
        status = "✅ GOOD"
    elif success_rate >= 50 or coverage_rate >= 40:
        status = "👍 OK"
    else:
        status = "⚠️  POOR"
    
    print(f"   Tolerance: {complexity_within_tolerance}/{total_complexity_bins} ({success_rate:.0f}%) within ±{config['tolerance']*100:.0f}%")
    print(f"   Coverage: {high_coverage_complexity}/{len(complexity_batch_map)} ({coverage_rate:.0f}%) high-coverage bins")
    print(f"   Status: {status}")
    
    # Top bins summary
    top_complexity = sorted(complexity_targets.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"   Top targets: {', '.join([f'{name}({count})' for name, count in top_complexity])}")
    
    print()


def save_distribution_report(summary, subsets, output_file="analyze/sampler_distribution_report.json"):
    """
    Save detailed distribution report for offline analysis.
    
    Args:
        summary: Sampler summary from get_sampling_summary()
        subsets: List of problem family subsets
        output_file: Output file path for the report
    """
    # Build comprehensive report
    report = {
        "configuration": summary['configuration'],
        "available_options": summary['available_options'],
        "targets": summary['targets'],
        "batch_distribution": []
    }
    
    # Add batch details
    for i, subset in enumerate(subsets):
        meta = subset['batch_metadata']
        batch_info = {
            "batch_id": i + 1,
            "selected_complexity": meta['selected_complexity'],
            "selected_task_types": meta['selected_task_types'],
            "complexity_count": len(meta['selected_complexity']),
            "task_type_count": len(meta['selected_task_types'])
        }
        report["batch_distribution"].append(batch_info)
    
    # Calculate coverage statistics
    complexity_coverage = defaultdict(list)
    task_type_coverage = defaultdict(list)
    
    for i, subset in enumerate(subsets):
        meta = subset['batch_metadata']
        for comp in meta['selected_complexity']:
            complexity_coverage[comp].append(i + 1)
        for task in meta['selected_task_types']:
            task_type_coverage[task].append(i + 1)
    
    report["coverage_analysis"] = {
        "complexity_coverage": dict(complexity_coverage),
        "task_type_coverage": dict(task_type_coverage)
    }
    
    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Distribution report saved to: {output_file}")


def main():
    """
    Main test function demonstrating sampler visualization.
    """
    print("🚀 Sampler Distribution Test")
    print("=" * 40)
    
    # Test scenarios with different configurations
    test_scenarios = [
        ("easy", 1, 10),      # Small: 10 samples
        ("medium", 5, 20),    # Medium: 100 samples  
        ("hard", 10, 100),    # Large: 1000 samples
    ]
    
    for difficulty, batch_size, num_iteration in test_scenarios:
        # Create sampler for this scenario
        histogram, sampler = create_sampler_from_config(
            problem_family_file="problem_family.json",
            difficulty=difficulty,
            batch_size=batch_size,
            num_iteration=num_iteration,
            tolerance=0.5,
            seed=42
        )
        
        # Create problem family subsets
        subsets = sampler.create_problem_family_subsets()
        
        # Visualize the distribution
        visualize_bin_distribution(histogram, sampler, subsets)
    
    # Save detailed report for the last scenario  
    print("📝 Saving detailed report...")
    summary = sampler.get_sampling_summary()
    save_distribution_report(summary, subsets)
    
    print("✅ Test complete! Check analyze/sampler_distribution_report.json for details.")


if __name__ == "__main__":
    main()
