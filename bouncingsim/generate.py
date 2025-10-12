import argparse
import json
import os
import sys
import random
import glob
import traceback  # Added: import traceback module
from typing import List, Dict, Any
from itertools import combinations  # Added
from loguru import logger

# Add project root to Python path (now we're in the root directory)
project_root = os.path.abspath(os.getcwd())
sys.path.insert(0, project_root)

try:
    from scene_generation.scene_generator import generate as generate_scenes, DIFF_NAMES
    from utils import generate_scene_messages, predict_ball_positions, check_distance
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def get_project_root():
    """Get the absolute path to the project root directory (ballsim folder)"""
    current_dir = os.path.abspath(os.getcwd())

    # If currently in the ballsim folder, return directly
    if os.path.basename(current_dir) == "ballsim":
        return current_dir

    # If currently in a subfolder of ballsim, search upwards
    while current_dir != os.path.dirname(current_dir):  # Avoid infinite loop
        if os.path.basename(current_dir) == "ballsim":
            return current_dir
        current_dir = os.path.dirname(current_dir)

    # If not found, use relative path
    return os.path.abspath(os.getcwd())

def check_directory_overwrite(dataset_name: str, scenes_dir: str, types_combinations: List[str], difficulties: List[int], scenes_per_config: int, skip_existing: bool = False):
    """Check if directories will be overwritten and provide warnings"""
    total_scenes = len(types_combinations) * len(difficulties) * scenes_per_config
    
    # Check each type and difficulty combination
    existing_scenes = 0
    overwrite_warnings = []
    
    for types_str in types_combinations:
        for difficulty in difficulties:
            difficulty_name = DIFF_NAMES[difficulty]
            scene_folder = os.path.join(scenes_dir, f"scenes_{types_str}", difficulty_name)
            
            if os.path.exists(scene_folder):
                # Calculate the number of existing scenes
                pattern = os.path.join(scene_folder, f"scene_{types_str}_{difficulty_name}_*.json")
                existing_files = glob.glob(pattern)
                existing_scenes += len(existing_files)
                
                if existing_files:
                    overwrite_warnings.append(f"  - {types_str}_d{difficulty}: {len(existing_files)} existing scenes")
    
    if existing_scenes > 0:
        if skip_existing:
            print(f"\n📋 INFO: Found {existing_scenes} existing scenes")
            print("   Will skip existing scenes and only generate missing ones.")
            print("   Use --skip-existing=false to overwrite existing scenes.\n")
        else:
            print(f"\n⚠️  WARNING: Dataset '{dataset_name}' will overwrite existing scenes!")
            print(f"   Total existing scenes to be overwritten: {existing_scenes}")
            print(f"   New scenes to be generated: {total_scenes}")
            print("   Affected directories:")
            for warning in overwrite_warnings:
                print(warning)
            
            response = input(f"\nDo you want to continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Dataset generation cancelled.")
                sys.exit(0)
            print("Continuing with dataset generation...\n")
    else:
        print(f"[OK] No existing scenes found. Will generate {total_scenes} new scenes.\n")

def generate_random_timestamps(count: int = 10, seed: int = None, difficulty: int = 0) -> List[float]:
    """
    Generate random timestamps for evaluation.
    
    Strategy (as documented in design.md):
    - Range varies by difficulty level
    - Distribution: Stratified random sampling
    - Precision: Round to 2 decimal places for consistency
    
    Difficulty-based timestamp ranges:
    - Basic (0): 0.5s to 3.0s (avoids too-early trivial cases and too-late chaotic behavior)
    - Easy (1): 0.5s to 4.0s (slightly extended for more complex dynamics)
    - Medium (2): 0.5s to 5.0s (extended range for intermediate complexity)
    - Hard (3): 0.5s to 8.0s (extended range for complex multi-collision scenarios)
    - Extreme (4): 2.0s to 20.0s (significantly extended time range to test long-term prediction and complex dynamics)
    
    Distribution strategy:
    - Early bias: 20-30% of samples in early range for easier cases
    - Late coverage: 70-80% in later range for complex dynamics
    """
    if seed is not None:
        random.seed(seed)
    
    # Set timestamp ranges based on difficulty
    difficulty_ranges = {
        0: (0.1, 2.1),    # Basic: 0.5s to 3.0s
        1: (0.1, 2.1),    # Easy: 0.5s to 4.0s
        2: (0.1, 2.1),    # Medium: 0.5s to 5.0s
        3: (0.1, 2.1),    # Hard: 0.5s to 8.0s
        4: (0.1, 2.1),   # Extreme: 2.0s to 20.0s (大幅增加时间范围)
    }
    
    min_time, max_time = difficulty_ranges.get(difficulty, (0.5, 3.0))
    
    # Adjust early and late time allocation based on difficulty
    if difficulty <= 1:  # Basic/Easy
        early_ratio = 0.3  # 30% early
        early_max = 1.0
    elif difficulty == 2:  # Medium
        early_ratio = 0.25  # 25% early
        early_max = 0.5
    elif difficulty == 3:  # Hard
        early_ratio = 0.2   # 20% early
        early_max = 2.0
    else:  # Extreme
        early_ratio = 0.15  # 15% early (reduced early time ratio)
        early_max = 4.0     # Early time extended to 4.0s
    
    timestamps = []
    
    # Early timestamps for simpler dynamics
    early_count = int(count * early_ratio)
    for _ in range(early_count):
        if difficulty == 4:  # Extreme: early time starts from 2.0s
            t = random.uniform(2.0, early_max)
        else:
            t = random.uniform(0.5, early_max)
        timestamps.append(round(t, 2))
    
    # Later timestamps for complex dynamics  
    late_count = count - early_count
    for _ in range(late_count):
        t = random.uniform(early_max, max_time)
        timestamps.append(round(t, 2))
    
    # Sort for consistent ordering
    timestamps.sort()
    return timestamps

def generate_periodic_timestamps(interval: float = 1.0, count: int = 10, difficulty: int = 0) -> List[float]:
    """
    Generate strict periodic node timestamps for H scene types

    Args:
        interval: timestamp interval (default: 1.0s)
        count: number of timestamps
        difficulty: difficulty level

    Returns:
        List[float]: timestamps with interval spacing [0, interval, 2*interval, ..., (count-1)*interval]
    """
    timestamps = []

    # Generate timestamps with interval spacing: 0, interval, 2*interval, ..., (count-1)*interval
    for cycle in range(count):
        cycle_time = cycle * interval + interval
        timestamps.append(round(cycle_time, 2))

    return timestamps

def generate_ground_truth_assertions(timestamps: List[float], ground_truth_positions: List[List[tuple]], tolerance: float = 50.0, prompt_setting: str = "list") -> List[str]:
    """
    Generate assertion strings for ground truth validation.
    
    Args:
        timestamps: List of timestamp values
        ground_truth_positions: List[List[tuple]] where outer list is per timestamp,
                               inner list is per ball, tuple is (x, y) position
        tolerance: Error tolerance in pixels (default: 50.0)
        prompt_setting: 'list' for predict_position() returning nested lists; 'function' for predict_position(t)
    
    Returns:
        List of assertion strings
    """
    assertions = []
    
    for i, (timestamp, positions) in enumerate(zip(timestamps, ground_truth_positions)):
        # Fixed: maintain float format, do not convert to integer
        position_list = [[round(float(x), 2), round(float(y), 2)] for x, y in positions]
        position_str = str(position_list)
        
        if prompt_setting == "function":
            assertion = f"assert check_distance(predict_position({float(timestamp):.2f}), {position_str}) <= {tolerance}"
        else:
            assertion = f"assert check_distance(predict_position()[{i}], {position_str}) <= {tolerance}"
        assertions.append(assertion)
    
    return assertions

def create_dataset_entry(scene_config: Dict[str, Any], scene_id: str, difficulty: int, 
                        timestamps: List[float], messages: List[Dict[str, str]], 
                        ground_truth_positions: List[List[tuple]], tolerance: float = 50.0, prompt_setting: str = "list") -> Dict[str, Any]:
    """Create a single JSONL entry for the dataset."""
    
    ground_truth_assertions = generate_ground_truth_assertions(timestamps, ground_truth_positions, tolerance, prompt_setting)
    
    return {
        "messages": messages,
        "ground_truth": ground_truth_assertions,
        "dataset": "ballsim",
        "id": scene_id,
        "difficulty": difficulty,
        "timestamps": timestamps,  # Additional metadata for analysis
        # "scene_config": scene_config,  # Additional metadata for debugging
        "tolerance": tolerance,  # Added: record tolerance
        "prompt_setting": prompt_setting  # Added: record prompt setting
    }

def main():
    parser = argparse.ArgumentParser(description="Generate JSONL dataset for polygon dynamics 2D problems.")
    
    # Scene generator parameters - Modified: support list input
    parser.add_argument("--types",  nargs="*", default=["AH"], help="Scene types to generate: single letters (A B), combinations (ABC DEF), or 'all' for all combinations")
    parser.add_argument("--dataset-name", type=str, default="scene_AH", help="Dataset name for output file")

    parser.add_argument("--difficulties", nargs="+", type=int, default=[4],
                       help="Difficulty levels to generate: 0=Basic, 1=Easy, 2=Medium, 3=Hard, 4=Extreme")
    parser.add_argument("--scenes-per-config", type=int, default=5,
                       help="Number of scenes per (type, difficulty) combination")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    parser.add_argument("--timestamps-per-scene", type=int, default=15,
                       help="Number of random timestamps per scene")
    parser.add_argument("--periodic-interval", type=float, default=0.1,
                       help="Interval between periodic timestamps for H scenes (default: 1.0s)")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                       help="Output directory for dataset files")
    parser.add_argument("--tolerance", type=float, default=50.0,
                       help="Error tolerance in pixels for ground truth assertions (default: 50.0)")
    parser.add_argument("--separate-by-difficulty", action="store_true", help="Generate separate dataset files for each difficulty level instead of one combined file")
    parser.add_argument("--prompt-setting", choices=["list", "function"], default="function",
                       help="Prompt style: 'list' reveals timestamps and expects nested-list output; 'function' hides timestamps and requires predict_position(t).")
    
    # Added: validation parameters
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip scene validation (faster but less safe)")
    parser.add_argument("--max-retries", type=int, default=500,
                       help="Maximum retries for failed scene generation")
    parser.add_argument("--skip-existing", action="store_true", help="Skip generating scenes that already exist")
    
    args = parser.parse_args()
    
    # Parameter validation
    if not all(0 <= d <= 4 for d in args.difficulties):
        print("Error: Difficulty levels must be 0-4")
        sys.exit(1)
    
    if args.scenes_per_config <= 0 or args.timestamps_per_scene <= 0:
        print("Error: scenes-per-config and timestamps-per-scene must be positive")
        sys.exit(1)
    
    # Set random seed
    random.seed(args.seed)
    
    # Get project root directory (balls folder)
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Create output directory - Modified: always at the balls level
    if args.output_dir.startswith("data/"):
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = os.path.join(project_root, "data", "datasets")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scene directory - Modified: create separate directory based on dataset name
    scenes_dir = os.path.join(project_root, "data", "scenes", args.dataset_name)
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Process types parameter - Modified: use new parsing function
    def parse_types_parameter(types_input):
        """Parse types parameter, support multiple formats"""
        if not types_input:
            return ["A"]
        
        # Check if "all" is included
        if "all" in [t.lower() for t in types_input]:
            print("[INFO] Found 'all' in types - generating dataset for ALL scene type combinations...")
            type_combinations = []
            # Single-axis scenes
            for letter in "ABCDEFGH":  # Added H scene type
                type_combinations.append(letter)
            # Combination scenes (all combinations of 2-8 axes)
            for combo_size in range(2, 9):
                for combo in combinations("ABCDEFGH", combo_size):  # Added H scene type
                    type_combinations.append("".join(combo))
            return type_combinations
        
        # Process explicitly specified types
        valid_letters = set("ABCDEFGH")  # Added H scene type
        type_combinations = []
        
        for type_str in types_input:
            # Clean and validate input
            clean_type = "".join(sorted(set(type_str.upper())))
            
            # Validate that all characters are valid letters
            if not set(clean_type).issubset(valid_letters):
                invalid = set(clean_type) - valid_letters
                print(f"Error: Invalid letters in type '{type_str}': {invalid}. Use A-H letters.")
                sys.exit(1)
            
            if clean_type and clean_type not in type_combinations:
                type_combinations.append(clean_type)
        
        if not type_combinations:
            return ["A"]
        
        return type_combinations
    
    type_combinations = parse_types_parameter(args.types)
    
    # Generate output filename(s)
    if args.separate_by_difficulty:
        output_files = {}
        for difficulty in args.difficulties:
            difficulty_name = DIFF_NAMES[difficulty]
            output_files[difficulty] = os.path.join(output_dir, f"dataset_{args.dataset_name}_{difficulty_name}.jsonl")
    else:
        output_file = os.path.join(output_dir, f"dataset_{args.dataset_name}.jsonl")
    
    # Check directory overwrite and provide warnings
    check_directory_overwrite(args.dataset_name, scenes_dir, type_combinations, args.difficulties, args.scenes_per_config, args.skip_existing)
    
    # Improved: more detailed progress information
    total_configs = len(type_combinations) * len(args.difficulties)
    total_expected_scenes = total_configs * args.scenes_per_config
    
    print(f"=== Dataset Generation Plan ===")
    print(f"Dataset name: {args.dataset_name}")
    if args.separate_by_difficulty:
        print(f"Output files per difficulty will be created in: {output_dir}")
    else:
        print(f"Output file: {output_file}")
    print(f"Scenes directory: {scenes_dir}")
    print(f"Scene type combinations: {len(type_combinations)} ({type_combinations[:5]}{'...' if len(type_combinations) > 5 else ''})")
    print(f"Difficulty levels: {args.difficulties}")
    print(f"Scenes per config: {args.scenes_per_config}")
    print(f"Timestamps per scene: {args.timestamps_per_scene}")
    print(f"Periodic interval: {args.periodic_interval}s")
    print(f"Prompt setting: {args.prompt_setting}")
    print(f"Error tolerance: {args.tolerance} pixels")  # Added: display tolerance
    print(f"Expected total scenes: {total_expected_scenes}")
    print(f"Expected total entries: {total_expected_scenes}")
    print(f"Random seed: {args.seed}")
    print("=" * 35)
    
    total_entries = 0
    failed_configs = []
    
    # Choose file handling method based on mode
    if args.separate_by_difficulty:
        # Generate separately by difficulty: create file handles for each difficulty
        file_handles = {}
        for difficulty in args.difficulties:
            file_handles[difficulty] = open(output_files[difficulty], 'w', encoding='utf-8')
    else:
        # Merged mode: create single file handle
        main_file = open(output_file, 'w', encoding='utf-8')
    
    # Generate data
    for config_idx, types_str in enumerate(type_combinations):
        for difficulty in args.difficulties:
            config_name = f"{types_str}_d{difficulty}"
            print(f"\n[{config_idx + 1}/{len(type_combinations)}] Generating {config_name}...")
            
            # Use scene_generator.generate() function to generate scene files
            difficulty_name = DIFF_NAMES[difficulty]
            
            # Call generation function (pass complete types string and dataset-specific scene directory)
            # try:
            generate_scenes(types_str, difficulty, args.scenes_per_config, scenes_dir, args.skip_existing, args.max_retries)
            # except Exception as e:
            #     print(f"  ERROR: Scene generation failed for {config_name}: {e}")
            #     failed_configs.append((config_name, str(e)))
            #     continue

            # Find scene files to process - only process specified number of scenes
            scene_folder = os.path.join(scenes_dir, f"scenes_{types_str}", difficulty_name)
            
            # Build list of specified number of scene files
            expected_scene_files = []
            for i in range(1, args.scenes_per_config + 1):
                scene_filename = f"scene_{types_str}_{difficulty_name}_{i}.json"
                scene_path = os.path.join(scene_folder, scene_filename)
                if os.path.exists(scene_path):
                    expected_scene_files.append(scene_path)
                else:
                    print(f"  WARNING: Expected scene file not found: {scene_filename}")
            
            if not expected_scene_files:
                warning_msg = f"No expected scene files found for {config_name} (expected 1-{args.scenes_per_config})"
                print(f"  WARNING: {warning_msg}")
                failed_configs.append((config_name, warning_msg))
                continue
            
            print(f"  Found {len(expected_scene_files)}/{args.scenes_per_config} expected scenes for {config_name}")
            
            config_entries = 0
            for scene_idx, scene_file in enumerate(expected_scene_files):
                # Parse index directly from filename: scene_X_difficulty_Y.json -> Y
                scene_filename = os.path.basename(scene_file)
                # Example: scene_D_easy_10.json -> ['scene', 'D', 'easy', '10', 'json']
                parts = scene_filename.replace('.json', '').split('_')
                real_index = int(parts[-1])  # Get last part as real index
                scene_id = f"{types_str}_{difficulty}_{real_index}"
                print(f"  Processing scene {scene_id} ({scene_idx+1}/{len(expected_scene_files)})...")
                
                # Load scene configuration
                try:
                    with open(scene_file, 'r', encoding='utf-8') as sf:
                        scene_config = json.load(sf)
                except Exception as e:
                    print(f"    ERROR: Failed to load scene file {scene_file}: {e}")
                    continue
                
                # Construct scene_path (relative path, for predict.py use)
                # Modified: construct scene path with complete directory information
                # Format: dataset_name/scenes_type/difficulty/scene_name (without .json extension)
                difficulty_name = DIFF_NAMES[difficulty]
                scene_filename = os.path.basename(scene_file)
                scene_name_without_ext = os.path.splitext(scene_filename)[0]

                # Construct complete scene path: dataset_name/scenes_type/difficulty/scene_name
                scene_path = f"{args.dataset_name}/scenes_{types_str}/{difficulty_name}/{scene_name_without_ext}"
                
                print(f"    Scene path for predict.py: {scene_path}")
                
                # Check if scene has pre-computed test cases
                if "tests" in scene_config and scene_config["tests"]:
                    print(f"    Using pre-computed test cases from scene file ({len(scene_config['tests'])} tests)")
                    # Extract timestamps and positions from tests field
                    timestamps = [test["time"] for test in scene_config["tests"]]
                    ground_truth_positions = [test["position"] for test in scene_config["tests"]]
                    simulation_failed = False
                else:
                    # Generate timestamps based on scene type
                    if "H" in types_str and "periodic_motion" in scene_config["meta"]:
                        # H scene type: use periodic timestamps
                        period = scene_config["meta"]["periodic_motion"]["period"]
                        print(f"    H scene detected, using periodic timestamps with interval {args.periodic_interval:.2f}s")
                        timestamps = generate_periodic_timestamps(
                            interval=args.periodic_interval,
                            count=args.timestamps_per_scene,
                            difficulty=difficulty
                        )
                    else:
                        # Other scene types: use random timestamps
                        # timestamps = generate_random_timestamps(
                        #     count=args.timestamps_per_scene,
                        #     seed=args.seed + total_entries,  # Unique seed per scene
                        #     difficulty=difficulty  # 传递难度
                        # )
                        timestamps = generate_periodic_timestamps(
                            interval=args.periodic_interval,
                            count=args.timestamps_per_scene,
                            difficulty=difficulty
                        )
                    
                    # Simulate ground truth for each timestamp
                    ground_truth_positions = []
                    simulation_failed = False
                    
                    for timestamp_idx, timestamp in enumerate(timestamps):
                        try:
                            # Fixed: pass correct scene_name and dataset_name parameters
                            # scene_name should be "scenes_type/difficulty/scene_name" format, dataset_name is dataset name
                            scene_name_for_predict = f"scenes_{types_str}/{difficulty_name}/{scene_name_without_ext}"
                            positions = predict_ball_positions(scene_name_for_predict, timestamp, args.dataset_name)
                            logger.info(f"try to scene_name_for_predict: {scene_name_for_predict}")
                            # logger.info(f"positions: {positions}")
                            ground_truth_positions.append(positions)
                            
                            # Added: validate reasonableness of prediction results
                            if not positions or not all(isinstance(pos, (list, tuple)) and len(pos) >= 2 for pos in positions):
                                print(f"    WARNING: Invalid position format for {scene_id} at t={timestamp:.2f}s: {positions}")
                                simulation_failed = True
                                break
                                
                        except Exception as e:
                            print(f"    ERROR: Simulation failed for {scene_id} at t={timestamp:.2f}s: {e}")
                            print(f"    TRACEBACK:")
                            traceback.print_exc()  # Added: print complete traceback
                            simulation_failed = True
                            break
                    
                    if simulation_failed:
                        continue
                
                # Generate prompt messages (Fixed: pass all timestamps)
                try:
                    messages = generate_scene_messages(
                        scene_config,
                        timestamps,  # Fixed: pass complete timestamps list
                        include_vertices=False,
                        prompt_setting=args.prompt_setting
                    )
                except Exception as e:
                    print(f"    ERROR: Failed to generate prompt for {scene_id}: {e}")
                    continue
                
                # Create dataset entry
                entry = create_dataset_entry(
                    scene_config=scene_config,
                    scene_id=scene_id,
                    difficulty=difficulty,
                    timestamps=timestamps,
                    messages=messages,
                    ground_truth_positions=ground_truth_positions,
                    tolerance=args.tolerance,  # Added: pass tolerance
                    prompt_setting=args.prompt_setting
                )
                
                # Write to JSONL file
                if args.separate_by_difficulty:
                    file_handles[difficulty].write(json.dumps(entry, ensure_ascii=False) + '\n')
                else:
                    # Merged mode: write to single file
                    main_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                total_entries += 1
                config_entries += 1
            
            print(f"  Completed {config_name}: {config_entries} entries generated")
    
    # Close file handles
    if args.separate_by_difficulty:
        for handle in file_handles.values():
            handle.close()
    else:
        main_file.close()
        
    print(f"\n=== Dataset Generation Completed ===")
    print(f"Total entries: {total_entries}")
    if failed_configs:
        print(f"Failed configs: {len(failed_configs)}")
        for cfg, err in failed_configs:
            print(f"  - {cfg}: {err}")
    else:
        print("All configurations completed successfully.")

if __name__ == "__main__":
    main() 