"""
Centralized Configuration for Problem Generators

This module loads configuration from YAML and provides access to all 
hyperparameters and settings used across different problem generators.
"""

import yaml
import os
from typing import List, Dict, Any


class GeneratorConfig:
    """Centralized configuration for all problem generators."""
    
    _config = None
    _config_file = "config.yaml"
    
    @classmethod
    def load_config(cls, config_file: str = None):
        """Load configuration from YAML file."""
        if config_file:
            cls._config_file = config_file
        
        try:
            with open(cls._config_file, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file '{cls._config_file}' not found. Using defaults.")
            cls._config = cls._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            cls._config = cls._get_default_config()
        
        # Populate class attributes from loaded config
        cls._populate_attributes()
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration if YAML file is not available."""
        return {
            'sequence_lengths': {
                'starts_with': [1, 6],
                'exact_sequence': [1, 6],
                'contains_ordered': [1, 6],
                'contains_substring': [1, 6],
                'ends_with': [1, 6]
            },
            'count_thresholds': 11,
            'numerical_thresholds': [1, 256],
            'numerical_operations': ['even', 'odd', 'power_of_2', 'greater_than'],
            'regex': {
                'operators': ['+', '?', '*'],
                'concatenation_count': 2,
                'max_pattern_length': [1, 3]
            },
            'test_cases': {
                'max_test_cases': 35,
                'min_accepting_cases': 3,
                'min_rejecting_cases': 3,
                'string_min_length': 1,
                'string_max_length': 12,
                'complex_case_min_length': 8,
                'complex_cases_per_pattern': 6
            },
            'complex_case_lengths': {
                'short_complex': 10,
                'medium_complex': 12,
                'long_complex': 15,
                'large_complex': 18,
                'very_long_complex': 20
            },
            'random_padding_lengths': {
                'short': [3, 5],
                'medium': [6, 8],
                'long': [9, 12],
                'very_long': [13, 20]
            },
            'generation': {
                'max_generation_attempts': 50,
                'complexity_progression_factor': 10,
                'starting_problem_id': 1000,
                'mode': 'systematic'
            },
            'patterns': {
                'alternating_long_pattern_length': 16,
                'balanced_colors_per_count': [3, 4, 6, 8],
                'max_binary_string_length': 1024
            },
            'colors': {
                'two_color': ['R', 'B'],
                'four_color': ['R', 'B', 'Y', 'G']
            },
            'difficulty_weights': {
                'basic': 0.3,
                'easy': 0.3,
                'medium': 0.2,
                'hard': 0.2
            },
            'validation': {
                'max_string_length': 50,
                'corner_case_max_length': 3,
                'corner_case_max_size': 5
            }
        }
    
    @classmethod
    def get(cls, key: str, default=None):
        """Get configuration value by key."""
        if cls._config is None:
            cls.load_config()
        
        keys = key.split('.')
        value = cls._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @classmethod
    def _populate_attributes(cls):
        """Populate class attributes from loaded config."""
        # Sequence lengths
        cls.STARTS_WITH_SEQ_LENGTH = tuple(cls.get('sequence_lengths.starts_with', [1, 6]))
        cls.EXACT_SEQUENCE_LENGTH = tuple(cls.get('sequence_lengths.exact_sequence', [1, 6]))
        cls.CONTAINS_ORDERED_LENGTH = tuple(cls.get('sequence_lengths.contains_ordered', [1, 6]))
        cls.CONTAINS_SUBSTRING_LENGTH = tuple(cls.get('sequence_lengths.contains_substring', [1, 6]))
        cls.ENDS_WITH_SEQ_LENGTH = tuple(cls.get('sequence_lengths.ends_with', [1, 6]))
        cls.APPEND_SEQUENCE_LENGTH = tuple(cls.get('sequence_lengths.append_sequence', [1, 6]))
        cls.PREPEND_SEQUENCE_LENGTH = tuple(cls.get('sequence_lengths.prepend_sequence', [1, 6]))
        cls.PREPEND_ENABLE_MUTATIONS = cls.get('prepend_sequence.enable_mutations', True)
        cls.PREPEND_PATTERN_MUTATIONS = cls.get('prepend_sequence.pattern_mutations', False)
        
        # Count and numerical parameters
        count_thresholds = cls.get('count_thresholds', 11)
        # Support both single value and range [min, max]
        if isinstance(count_thresholds, (list, tuple)):
            cls.COUNT_THRESHOLDS = tuple(count_thresholds)
        else:
            cls.COUNT_THRESHOLDS = count_thresholds
        cls.NUMERICAL_THRESHOLDS = tuple(cls.get('numerical_thresholds', [1, 256]))
        cls.NUMERICAL_OPERATIONS = cls.get('numerical_operations', ['even', 'odd', 'power_of_2', 'greater_than'])
        
        # Numerical operations parameters
        cls.ARITHMETIC_OPERATIONS = cls.get('numerical_operations_config.arithmetic_operations', ['add', 'subtract'])
        cls.BITWISE_OPERATIONS = cls.get('numerical_operations_config.bitwise_operations', ['and', 'or', 'xor', 'not'])
        cls.SHIFT_OPERATIONS = cls.get('numerical_operations_config.shift_operations', ['floor_div'])
        cls.OPERAND_VALUES = cls.get('numerical_operations_config.operand_values', [1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 7, 15, 31, 63, 127])
        cls.MAX_SHIFT_POWER = cls.get('numerical_operations_config.max_shift_power', 7)
        
        # Max/Min operations parameters
        cls.MAX_MIN_OPERATIONS = cls.get('max_min_operations_config.operations', ['max', 'min'])
        cls.MAX_MIN_OPERAND_RANGE = tuple(cls.get('max_min_operations_config.operand_range', [1, 256]))
        
        # Regex parameters
        cls.REGEX_OPERATORS = cls.get('regex.operators', ['+', '?', '*'])
        cls.REGEX_CONCATENATION_COUNT = cls.get('regex.concatenation_count', 2)
        # Support both single value and range for max_pattern_length
        max_pattern_length_config = cls.get('regex.max_pattern_length', [1, 3])
        if isinstance(max_pattern_length_config, (list, tuple)):
            cls.REGEX_MAX_PATTERN_LENGTH = tuple(max_pattern_length_config)
        else:
            # Backwards compatibility: single value becomes range [1, value]
            cls.REGEX_MAX_PATTERN_LENGTH = (1, max_pattern_length_config)
        
        # Complex case lengths
        cls.COMPLEX_CASE_LENGTHS = cls.get('complex_case_lengths', {
            'short_complex': 10,
            'medium_complex': 12,
            'long_complex': 15,
            'large_complex': 18,
            'very_long_complex': 20
        })
        
        # Test case parameters
        cls.MAX_TEST_CASES = cls.get('test_cases.max_test_cases', 35)
        cls.MIN_ACCEPTING_CASES = cls.get('test_cases.min_accepting_cases', 3)
        cls.MIN_REJECTING_CASES = cls.get('test_cases.min_rejecting_cases', 3)
        cls.TEST_STRING_MIN_LENGTH = cls.get('test_cases.string_min_length', 1)
        cls.TEST_STRING_MAX_LENGTH = cls.get('test_cases.string_max_length', 12)
        cls.COMPLEX_CASE_MIN_LENGTH = cls.get('test_cases.complex_case_min_length', 8)
        cls.COMPLEX_CASES_PER_PATTERN = cls.get('test_cases.complex_cases_per_pattern', 6)
        
        # Complex case and padding parameters
        cls.COMPLEX_CASE_LENGTHS = cls.get('complex_case_lengths', {
            'short_complex': 10,
            'medium_complex': 12,
            'long_complex': 15,
            'large_complex': 18,
            'very_long_complex': 20
        })
        cls.RANDOM_PADDING_LENGTHS = cls.get('random_padding_lengths', {
            'short': [3, 5],
            'medium': [6, 8],
            'long': [9, 12],
            'very_long': [13, 20]
        })
        
        # Generation parameters
        cls.MAX_GENERATION_ATTEMPTS = cls.get('generation.max_generation_attempts', 50)
        cls.COMPLEXITY_PROGRESSION_FACTOR = cls.get('generation.complexity_progression_factor', 10)
        cls.STARTING_PROBLEM_ID = cls.get('generation.starting_problem_id', 1000)
        cls.GENERATION_MODE = cls.get('generation.mode', 'systematic')  # 'systematic' or 'random'
        
        # Pattern-specific parameters
        cls.ALTERNATING_LONG_PATTERN_LENGTH = cls.get('patterns.alternating_long_pattern_length', 16)
        cls.BALANCED_COLORS_PER_COUNT = cls.get('patterns.balanced_colors_per_count', [3, 4, 6, 8])
        cls.MAX_BINARY_STRING_LENGTH = cls.get('patterns.max_binary_string_length', 1024)
        
        # Color parameters
        cls.TWO_COLOR_CHARS = cls.get('colors.two_color', ['R', 'B'])
        cls.FOUR_COLOR_CHARS = cls.get('colors.four_color', ['R', 'B', 'Y', 'G'])
        
        # Difficulty and validation parameters
        cls.DIFFICULTY_WEIGHTS = cls.get('difficulty_weights', {
            'basic': 0.3,
            'easy': 0.3,
            'medium': 0.2,
            'hard': 0.2
        })
        cls.MAX_VALIDATION_STRING_LENGTH = cls.get('validation.max_string_length', 50)
        cls.CORNER_CASE_MAX_LENGTH = cls.get('validation.corner_case_max_length', 3)
        cls.CORNER_CASE_MAX_SIZE = cls.get('validation.corner_case_max_size', 5)
    
    @classmethod
    def get_complex_case_length(cls, length_type: str) -> int:
        """Get complex case length by type."""
        return cls.COMPLEX_CASE_LENGTHS.get(length_type, 12)
    
    @classmethod
    def get_sequence_length_range(cls, pattern_type: str):
        """Get sequence length range for a specific pattern type."""
        if cls._config is None:
            cls.load_config()
        
        # Map pattern types to their config attributes
        pattern_mapping = {
            'starts_with': cls.STARTS_WITH_SEQ_LENGTH,
            'ends_with': cls.ENDS_WITH_SEQ_LENGTH,
            'exact_sequence': cls.EXACT_SEQUENCE_LENGTH,
            'contains_ordered': cls.CONTAINS_ORDERED_LENGTH,
            'contains_substring': cls.CONTAINS_SUBSTRING_LENGTH,
            'append_sequence': cls.APPEND_SEQUENCE_LENGTH,
            'prepend_sequence': cls.PREPEND_SEQUENCE_LENGTH
        }
        
        if pattern_type in pattern_mapping:
            return pattern_mapping[pattern_type]
        else:
            # Default fallback
            return (1, 6)
    
    @classmethod
    def get_random_padding_length(cls, length_type: str):
        """Get random padding length range for a specific length type."""
        if cls._config is None:
            cls.load_config()
        
        if length_type in cls.RANDOM_PADDING_LENGTHS:
            return tuple(cls.RANDOM_PADDING_LENGTHS[length_type])
        else:
            # Default fallback for unknown length types
            return (3, 5)


# Load configuration on import
GeneratorConfig.load_config() 