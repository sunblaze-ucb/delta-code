#!/usr/bin/env python3
"""
Difficulty-Level Problem Generator

Generates problems at different difficulty levels (basic, easy, medium, hard) 
with appropriate parameter configurations for each level.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from itertools import product
import copy
import yaml

# Import the registry system
from manufactoria_problem_generators import GeneratorConfig, GeneratorRegistry


class DynamicConfig:
    """A dynamic configuration class that can override GeneratorConfig parameters."""
    
    def __init__(self, overrides: Dict[str, Any] = None):
        """Initialize with optional parameter overrides."""
        self.overrides = overrides or {}
        
        # Create a copy of the original config values for restoration
        self.original_values = {}
    
    def __enter__(self):
        """Context manager entry - apply overrides."""
        # Store original values and apply overrides
        for key, value in self.overrides.items():
            if hasattr(GeneratorConfig, key):
                self.original_values[key] = getattr(GeneratorConfig, key)
                setattr(GeneratorConfig, key, value)
            else:
                print(f"Warning: Config parameter '{key}' not found in GeneratorConfig")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore original values."""
        # Restore original values
        for key, value in self.original_values.items():
            setattr(GeneratorConfig, key, value)


class DifficultyLevelGenerator:
    """Generator that creates problems at different difficulty levels."""
    
    def __init__(self, config_file: str = 'configs/manufactoria_config.yaml'):
        self.generators = GeneratorRegistry.get_all_generators()
        self.config_file = config_file
        self.config = self.load_config(config_file)
        
        # Load difficulty levels from config
        difficulty_section = self.config.get('difficulty', {})
        levels_config = difficulty_section.get('levels', {})
        self.difficulty_levels = list(levels_config.keys()) if levels_config else ['basic', 'easy', 'medium', 'hard']
    
    def get_available_pattern_types(self) -> List[str]:
        """Get all available pattern types."""
        return GeneratorRegistry.list_pattern_types()
    
    def get_difficulty_levels(self) -> List[str]:
        """Get all available difficulty levels."""
        return self.difficulty_levels
    
    def get_difficulty_info(self) -> Dict[str, Dict[str, Any]]:
        """Get difficulty level information including weights and descriptions."""
        difficulty_section = self.config.get('difficulty', {})
        return difficulty_section.get('levels', {})
    
    def get_max_problems_per_difficulty(self) -> int:
        """Get the global maximum problems per difficulty level."""
        difficulty_section = self.config.get('difficulty', {})
        return difficulty_section.get('max_problems_per_difficulty', 1000)
    
    def generate_problems_by_difficulty(self,
                                      pattern_type: str,
                                      difficulty_level: str,
                                      count: int = None,
                                      color_mode: str = 'two_color',
                                      output_dir: str = 'problems/difficulty',
                                      custom_config: Dict[str, Any] = None,
                                      generation_mode: str = None) -> List[Dict]:
        """
        Generate problems at a specific difficulty level.
        
        Args:
            pattern_type: The pattern type to generate (e.g., 'numerical_comparison')
            difficulty_level: The difficulty level ('basic', 'easy', 'medium', 'hard')
            count: Number of problems to generate (if None, uses global max_problems_per_difficulty)
            color_mode: 'two_color' or 'four_color'
            output_dir: Directory to save the problems
            custom_config: Optional custom configuration overrides
        
        Returns:
            List of generated problems
        """
        
        if pattern_type not in self.generators:
            available_types = ', '.join(self.get_available_pattern_types())
            raise ValueError(f"Unknown pattern type '{pattern_type}'. Available types: {available_types}")
        
        if difficulty_level not in self.difficulty_levels:
            raise ValueError(f"Unknown difficulty level '{difficulty_level}'. Available levels: {', '.join(self.difficulty_levels)}")
        
        # Use global cap if count is not specified
        if count is None:
            count = self.get_max_problems_per_difficulty()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        chars = (GeneratorConfig.TWO_COLOR_CHARS if color_mode == 'two_color' 
                else GeneratorConfig.FOUR_COLOR_CHARS)
        is_four_color = (color_mode == 'four_color')
        
        generator = self.generators[pattern_type]
        
        print(f"Generating {pattern_type} problems at {difficulty_level} difficulty...")
        print(f"  Count: {count} problems")
        
        # Get difficulty-specific configuration
        difficulty_config = self._get_difficulty_config(pattern_type, difficulty_level)
        
        # Apply custom config overrides if provided
        if custom_config:
            difficulty_config.update(custom_config)
        
        # Apply pattern-specific generation mode if provided
        if generation_mode:
            difficulty_config['generation_mode'] = generation_mode
        
        print(f"  Difficulty config: {difficulty_config}")
        
        # Prepare config overrides for DynamicConfig
        config_overrides = self._prepare_config_overrides(difficulty_config)
        
        problems = []
        
        with DynamicConfig(config_overrides):
            # Generate parameters
            params_list = generator.generate_parameters(chars, count)
            
            # Generate problems
            for i, params in enumerate(params_list):
                try:
                    problem = generator.generate_problem(
                        chars, is_four_color, params, i
                    )
                    problem['id'] = str(uuid.uuid4())
                    problem['index'] = i
                    problem['color_mode'] = color_mode
                    problem['difficulty_level'] = difficulty_level
                    problem['pattern_type'] = pattern_type
                    problems.append(problem)
                    
                except Exception as e:
                    print(f"Warning: Failed to generate problem {i+1}: {e}")
                    continue
        
        # Save problems to file
        self._save_problems(problems, pattern_type, difficulty_level, output_dir, color_mode)
        
        print(f"Successfully generated {len(problems)} problems at {difficulty_level} difficulty")
        
        return problems
    
    def _get_difficulty_config(self, pattern_type: str, difficulty_level: str) -> Dict[str, Any]:
        """Get configuration parameters for a specific pattern type and difficulty level from config."""
        
        difficulty_section = self.config.get('difficulty', {})
        parameter_mappings = difficulty_section.get('parameter_mappings', {})
        
        if pattern_type not in parameter_mappings:
            print(f"Warning: No difficulty mappings found for pattern type '{pattern_type}', using default")
            return {}
        
        pattern_difficulties = parameter_mappings[pattern_type]
        if difficulty_level not in pattern_difficulties:
            print(f"Warning: No difficulty config found for '{pattern_type}' at '{difficulty_level}' level, using default")
            return {}
        
        return pattern_difficulties[difficulty_level]
    
    def _prepare_config_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration overrides for DynamicConfig."""
        if config is None:
            return {}
        
        overrides = {}
        
        # Map config keys to GeneratorConfig attribute names
        config_mapping = {
            'numerical_thresholds': 'NUMERICAL_THRESHOLDS',
            'count_thresholds': 'COUNT_THRESHOLDS'
        }
        
        for config_key, value in config.items():
            if config_key in config_mapping:
                attr_name = config_mapping[config_key]
                overrides[attr_name] = value
            elif config_key == 'sequence_lengths':
                # Handle sequence_lengths specially - map to individual attributes
                if isinstance(value, dict):
                    for pattern_type, lengths in value.items():
                        if pattern_type == 'starts_with':
                            overrides['STARTS_WITH_SEQ_LENGTH'] = tuple(lengths) if isinstance(lengths, list) else lengths
                        elif pattern_type == 'ends_with':
                            overrides['ENDS_WITH_SEQ_LENGTH'] = tuple(lengths) if isinstance(lengths, list) else lengths
                        elif pattern_type == 'exact_sequence':
                            overrides['EXACT_SEQUENCE_LENGTH'] = tuple(lengths) if isinstance(lengths, list) else lengths
                        elif pattern_type == 'contains_ordered':
                            overrides['CONTAINS_ORDERED_LENGTH'] = tuple(lengths) if isinstance(lengths, list) else lengths
                        elif pattern_type == 'contains_substring':
                            overrides['CONTAINS_SUBSTRING_LENGTH'] = tuple(lengths) if isinstance(lengths, list) else lengths
                        elif pattern_type == 'append_sequence':
                            overrides['APPEND_SEQUENCE_LENGTH'] = tuple(lengths) if isinstance(lengths, list) else lengths
                        elif pattern_type == 'prepend_sequence':
                            overrides['PREPEND_SEQUENCE_LENGTH'] = tuple(lengths) if isinstance(lengths, list) else lengths
            elif config_key == 'regex':
                # Handle regex config specially
                if 'max_pattern_length' in value:
                    max_pattern_length = value['max_pattern_length']
                    # Support both single value and range for max_pattern_length
                    if isinstance(max_pattern_length, (list, tuple)):
                        overrides['REGEX_MAX_PATTERN_LENGTH'] = tuple(max_pattern_length)
                    else:
                        # Backwards compatibility: single value becomes range [1, value]
                        overrides['REGEX_MAX_PATTERN_LENGTH'] = (1, max_pattern_length)
                if 'concatenation_count' in value:
                    overrides['REGEX_CONCATENATION_COUNT'] = value['concatenation_count']
            elif config_key == 'generation_mode':
                # Handle pattern-specific generation mode override
                overrides['GENERATION_MODE'] = value
            elif config_key == 'prepend_sequence':
                # Handle prepend_sequence specific config
                if 'enable_mutations' in value:
                    overrides['PREPEND_ENABLE_MUTATIONS'] = value['enable_mutations']
                if 'pattern_mutations' in value:
                    overrides['PREPEND_PATTERN_MUTATIONS'] = value['pattern_mutations']
            elif config_key == 'numerical_operations_config':
                # Handle numerical operations config
                # For exclusive configuration: if any operation type is specified,
                # set unspecified types to empty lists
                specified_operation_types = []
                if 'arithmetic_operations' in value:
                    overrides['ARITHMETIC_OPERATIONS'] = value['arithmetic_operations']
                    specified_operation_types.append('arithmetic')
                if 'bitwise_operations' in value:
                    overrides['BITWISE_OPERATIONS'] = value['bitwise_operations']
                    specified_operation_types.append('bitwise')
                if 'shift_operations' in value:
                    overrides['SHIFT_OPERATIONS'] = value['shift_operations']
                    specified_operation_types.append('shift')
                
                # If at least one operation type is specified, set others to empty
                if specified_operation_types:
                    if 'arithmetic' not in specified_operation_types:
                        overrides['ARITHMETIC_OPERATIONS'] = []
                    if 'bitwise' not in specified_operation_types:
                        overrides['BITWISE_OPERATIONS'] = []
                    if 'shift' not in specified_operation_types:
                        overrides['SHIFT_OPERATIONS'] = []
                
                if 'operand_values' in value:
                    overrides['OPERAND_VALUES'] = value['operand_values']
                if 'max_shift_power' in value:
                    overrides['MAX_SHIFT_POWER'] = value['max_shift_power']
            elif config_key == 'max_min_operations_config':
                # Handle max/min operations config
                if 'operations' in value:
                    overrides['MAX_MIN_OPERATIONS'] = value['operations']
                if 'operand_range' in value:
                    overrides['MAX_MIN_OPERAND_RANGE'] = tuple(value['operand_range']) if isinstance(value['operand_range'], list) else value['operand_range']
        
        return overrides
    
    def _save_problems(self, problems: List[Dict], pattern_type: str, difficulty_level: str,
                      output_dir: str, color_mode: str):
        """Save problems to JSONL file."""
        
        if not problems:
            return
            
        # Create filename
        filename = f"{pattern_type}_{color_mode}_{difficulty_level}.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        # Save as JSONL format
        with open(filepath, 'w', encoding='utf-8') as f:
            for problem in problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(problems)} problems to {filepath}")
    
    def generate_multiple_difficulties(self,
                                     pattern_list: List[str] = None,
                                     output_dir: str = None) -> Dict[str, Dict]:
        """
        Generate problems for multiple pattern types and difficulty levels.
        
        Args:
            pattern_list: List of pattern names to generate (if None, uses active_patterns from config)
            output_dir: Directory to save all problems (if None, uses config default)
        
        Returns:
            Dictionary mapping pattern_type to difficulty results
        """
        
        difficulty_section = self.config.get('difficulty', {})
        
        if pattern_list is None:
            pattern_list = difficulty_section.get('active_patterns', [])
        
        if output_dir is None:
            output_dir = difficulty_section.get('output_dir', 'problems/difficulty')
        
        patterns = difficulty_section.get('patterns', {})
        defaults = difficulty_section.get('defaults', {})
        default_difficulties = defaults.get('difficulties', ['basic', 'easy', 'medium', 'hard'])
        default_color_mode = defaults.get('color_mode', 'two_color')
        
        all_results = {}
        
        for pattern_name in pattern_list:
            print(f"\n{'='*50}")
            print(f"Processing pattern: {pattern_name}")
            print(f"{'='*50}")
            
            if pattern_name not in patterns:
                print(f"Warning: Pattern '{pattern_name}' not found in configuration")
                continue
            
            pattern_config = patterns[pattern_name]
            pattern_results = {}
            
            difficulties = pattern_config.get('difficulties', default_difficulties)
            color_mode = pattern_config.get('color_mode', default_color_mode)
            custom_config = pattern_config.get('custom_config', {})
            generation_mode = pattern_config.get('generation_mode', None)  # Pattern-specific mode override
            
            for difficulty_level in difficulties:
                try:
                    problems = self.generate_problems_by_difficulty(
                        pattern_type=pattern_name,
                        difficulty_level=difficulty_level,
                        count=None,  # Use global cap
                        color_mode=color_mode,
                        output_dir=output_dir,
                        custom_config=custom_config,
                        generation_mode=generation_mode
                    )
                    pattern_results[difficulty_level] = problems
                    
                except Exception as e:
                    print(f"Error generating {pattern_name} at {difficulty_level} difficulty: {e}")
                    pattern_results[difficulty_level] = []
            
            all_results[pattern_name] = pattern_results
        
        return all_results
    
    def load_config(self, config_file: str = 'configs/manufactoria_config.yaml') -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_file}' not found.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            return {}
    
    def run_from_config(self, config_file: str = None) -> Dict[str, Dict]:
        """Run difficulty-based generation from YAML configuration."""
        if config_file and config_file != self.config_file:
            # If a different config file is specified, reload
            config = self.load_config(config_file)
        else:
            config = self.config
        
        if not config:
            print("No valid configuration found. Exiting.")
            return {}
        
        difficulty_section = config.get('difficulty', {})
        patterns = difficulty_section.get('patterns', {})
        active_patterns = difficulty_section.get('active_patterns', list(patterns.keys()))
        output_dir = difficulty_section.get('output_dir', 'problems/difficulty')
        
        print(f"Loading configuration from: {config_file or self.config_file}")
        print(f"Active patterns: {active_patterns}")
        print(f"Output directory: {output_dir}")
        
        all_results = {}
        
        # Get default settings
        defaults = difficulty_section.get('defaults', {})
        default_difficulties = defaults.get('difficulties', ['basic', 'easy', 'medium', 'hard'])
        default_color_mode = defaults.get('color_mode', 'two_color')
        
        for pattern_name in active_patterns:
            if pattern_name not in patterns:
                print(f"Warning: Pattern '{pattern_name}' not found in configuration")
                continue
            
            pattern_config = patterns[pattern_name]
            pattern_type = pattern_name  # Use the key as the pattern type
            
            print(f"\n{'='*60}")
            print(f"Running pattern: {pattern_name}")
            print(f"{'='*60}")
            
            try:
                pattern_results = {}
                # Use defaults, but allow pattern-specific overrides
                difficulties = pattern_config.get('difficulties', default_difficulties)
                color_mode = pattern_config.get('color_mode', default_color_mode)
                custom_config = pattern_config.get('custom_config', {})
                generation_mode = pattern_config.get('generation_mode', None)  # Pattern-specific mode override
                
                # If difficulties is a list, generate for each difficulty level
                if isinstance(difficulties, list):
                    for difficulty_level in difficulties:
                        problems = self.generate_problems_by_difficulty(
                            pattern_type=pattern_type,
                            difficulty_level=difficulty_level,
                            count=None,  # Use global cap
                            color_mode=color_mode,
                            output_dir=output_dir,
                            custom_config=custom_config,
                            generation_mode=generation_mode
                        )
                        pattern_results[difficulty_level] = problems
                # If difficulties is a dict (legacy format), handle it too
                elif isinstance(difficulties, dict):
                    for difficulty_level, diff_config in difficulties.items():
                        count = diff_config.get('count', None) if isinstance(diff_config, dict) else None
                        problems = self.generate_problems_by_difficulty(
                            pattern_type=pattern_type,
                            difficulty_level=difficulty_level,
                            count=count,
                            color_mode=color_mode,
                            output_dir=output_dir,
                            custom_config=custom_config,
                            generation_mode=generation_mode
                        )
                        pattern_results[difficulty_level] = problems
                
                all_results[pattern_name] = pattern_results
                
            except Exception as e:
                print(f"Error running pattern '{pattern_name}': {e}")
                all_results[pattern_name] = {}
        
        return all_results


def main():
    """Main function that runs difficulty-level problem generation from YAML configuration."""
    
    generator = DifficultyLevelGenerator()
    
    print("=== Difficulty-Level Problem Generator ===")
    print(f"Available pattern types: {generator.get_available_pattern_types()}")
    print(f"Available difficulty levels: {generator.get_difficulty_levels()}")
    print(f"Max problems per difficulty: {generator.get_max_problems_per_difficulty()}")
    print()
    
    # Run patterns from YAML configuration
    all_results = generator.run_from_config('configs/manufactoria_config.yaml')
    
    # Print summary
    print(f"\n{'='*60}")
    print("Generation Summary")
    print(f"{'='*60}")
    
    total_problems = 0
    
    for pattern_name, difficulties in all_results.items():
        pattern_total = 0
        print(f"\n{pattern_name}:")
        for difficulty_level, problems in difficulties.items():
            count = len(problems)
            pattern_total += count
            print(f"  {difficulty_level}: {count} problems")
        print(f"  Total: {pattern_total} problems")
        total_problems += pattern_total
    
    print(f"\nGrand Total: {total_problems} problems generated")
    print(f"Files saved in: problems/difficulty/")


if __name__ == "__main__":
    main()