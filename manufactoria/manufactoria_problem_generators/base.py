"""
Base class for all problem generators.

Contains shared functionality like test case generation, parameter space management, etc.
"""

import random
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
from itertools import product
from abc import ABC, abstractmethod
from .config import GeneratorConfig
import numpy as np

class BaseGenerator(ABC):
    """Base class for all problem type generators."""
    
    def __init__(self):
        self.two_color_chars = GeneratorConfig.TWO_COLOR_CHARS
        self.four_color_chars = GeneratorConfig.FOUR_COLOR_CHARS

    def generate_test_string(self, chars: List[str], min_len: int = None, max_len: int = None) -> str:
        """Generate a random string using given characters."""
        if min_len is None:
            min_len = GeneratorConfig.TEST_STRING_MIN_LENGTH
        if max_len is None:
            max_len = GeneratorConfig.TEST_STRING_MAX_LENGTH
        length = random.randint(min_len, max_len)
        return ''.join(random.choice(chars) for _ in range(length))

    def generate_corner_cases(self, chars: List[str]) -> List[str]:
        """Generate basic corner cases."""
        cases = [""]  # Empty string
        
        # Single characters
        for char in chars:
            cases.append(char)
        
        # Two character combinations
        for c1 in chars:
            for c2 in chars:
                if len(f"{c1}{c2}") <= GeneratorConfig.CORNER_CASE_MAX_LENGTH:  # Keep short
                    cases.append(f"{c1}{c2}")

        cases = list(np.random.choice(cases, GeneratorConfig.CORNER_CASE_MAX_SIZE))

        cases.extend(["RB", "BR", "RBR", "BRB", "RRB", "BBR", ""])
        
        return list(set(cases))  # Remove duplicates

    def create_test_cases(self, pattern_func: Callable[[str], bool], chars: List[str], 
                         check_output: bool = False, pattern_type: str = "unknown", 
                         pattern_info: Dict = None) -> List[Dict]:
        """Create comprehensive test cases for a pattern with systematic generation."""
        test_cases = []
        pattern_info = pattern_info or {}
        seen_inputs = set()  # Track seen inputs to avoid duplicates efficiently
        
        # Generate deterministic test cases based on pattern type
        deterministic_cases = self._generate_deterministic_test_cases(pattern_type, chars, pattern_info)
        
        # Add deterministic cases
        for test_input in deterministic_cases:
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_accepted = pattern_func(test_input)
                test_cases.append({
                    "input": test_input,
                    "expected_output": test_input if check_output else "",
                    "expected_accepted": expected_accepted,
                    "check_output": check_output,
                    "description": ""
                })
        
        # Ensure we have enough accepting and rejecting cases
        accepted_count = sum(1 for tc in test_cases if tc["expected_accepted"])
        rejected_count = len(test_cases) - accepted_count
        
        # Add more accepting cases if needed
        if accepted_count < GeneratorConfig.MIN_ACCEPTING_CASES:
            additional_cases = self._generate_additional_cases(
                pattern_func, chars, True, seen_inputs, 
                GeneratorConfig.MIN_ACCEPTING_CASES - accepted_count
            )
            test_cases.extend(additional_cases)
        
        # Add more rejecting cases if needed  
        if rejected_count < GeneratorConfig.MIN_REJECTING_CASES:
            additional_cases = self._generate_additional_cases(
                pattern_func, chars, False, seen_inputs, 
                GeneratorConfig.MIN_REJECTING_CASES - rejected_count
            )
            test_cases.extend(additional_cases)
        
        return test_cases[:GeneratorConfig.MAX_TEST_CASES]

    def _generate_deterministic_test_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate deterministic test cases based on pattern type."""
        cases = []
        
        # Always include basic cases
        cases.extend(["", chars[0], chars[0] + chars[1] if len(chars) > 1 else chars[0]])
        
        # Add corner cases
        cases.extend(self.generate_corner_cases(chars))
        
        # Add pattern-specific cases (including complex ones)
        cases.extend(self._get_pattern_specific_cases(pattern_type, chars, pattern_info))
        
        # Remove duplicates while preserving order
        unique_cases = []
        seen = set()
        for case in cases:
            if case not in seen:
                seen.add(case)
                unique_cases.append(case)
        
        return unique_cases
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Override in subclasses to provide pattern-specific test cases."""
        return []
    
    def _generate_additional_cases(self, pattern_func: Callable[[str], bool], chars: List[str], 
                                  target_result: bool, seen_inputs: set, count_needed: int) -> List[Dict]:
        """Generate additional test cases with target result (accepting or rejecting)."""
        additional_cases = []
        attempts = 0
        max_attempts = GeneratorConfig.MAX_GENERATION_ATTEMPTS
        
        while len(additional_cases) < count_needed and attempts < max_attempts:
            # Generate with increasing complexity
            complexity = (attempts // GeneratorConfig.COMPLEXITY_PROGRESSION_FACTOR) + 1
            max_len = min(GeneratorConfig.TEST_STRING_MIN_LENGTH + complexity, GeneratorConfig.TEST_STRING_MAX_LENGTH)
            test_input = self.generate_test_string(chars, GeneratorConfig.TEST_STRING_MIN_LENGTH, max_len)
            
            if test_input not in seen_inputs and pattern_func(test_input) == target_result:
                seen_inputs.add(test_input)
                additional_cases.append({
                    "input": test_input,
                    "expected_output": "",
                    "expected_accepted": target_result,
                    "check_output": False,
                    "description": ""
                })
            
            attempts += 1
        
        return additional_cases

    def create_problem_dict(self, problem_id: int, name: str, criteria: str, 
                           test_cases: List[Dict], difficulty: str, problem_type: str,
                           is_four_color: bool = False, description: str = "") -> Dict[str, Any]:
        """Create a standardized problem dictionary."""
        color_note = " The input tape carries only red/blue colors." if not is_four_color else ""
        
        # Define available nodes based on color mode and complexity
        if is_four_color:
            available_nodes = ["START", "END", "PULLER_RB", "PULLER_YG", 
                             "PAINTER_RED", "PAINTER_BLUE", "PAINTER_YELLOW", "PAINTER_GREEN"]
        else:
            if difficulty in ['basic', 'medium'] and problem_type in ['starts_with', 'exact_sequence', 'ends_with']:
                available_nodes = ["START", "END", "PULLER_RB"]
            else:
                available_nodes = ["START", "END", "PULLER_RB", "PULLER_YG", 
                                 "PAINTER_RED", "PAINTER_BLUE", "PAINTER_YELLOW", "PAINTER_GREEN"]
        
        return {
            "id": problem_id,
            "name": name,
            "description": description,
            "criteria": criteria + color_note,
            "available_nodes": available_nodes,
            "test_cases": test_cases,
            "solutions": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "difficulty": difficulty,
            "problem_type": problem_type
        }

    def get_sequence_space(self, chars: List[str], min_len: int, max_len: int) -> List[str]:
        """Generate all possible sequences of given length range."""
        sequences = []
        for length in range(min_len, max_len + 1):
            for seq in product(chars, repeat=length):
                sequences.append(''.join(seq))
        return sequences

    @abstractmethod
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a problem of this type. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate how many unique problems this pattern type can generate."""
        pass

    @abstractmethod
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for this pattern type."""
        pass

    @abstractmethod
    def get_difficulty(self) -> str:
        """Return the difficulty level for this pattern type."""
        pass

    @abstractmethod
    def get_pattern_type(self) -> str:
        """Return the pattern type identifier."""
        pass

    def uses_only_two_colors(self, params: Dict[str, Any]) -> bool:
        """Check if the given parameters would result in a problem that only uses R and B colors."""
        # Default implementation - subclasses can override for pattern-specific logic
        return False
    
    def should_generate_four_color_version(self, params: Dict[str, Any]) -> bool:
        """Determine if a four-color version should be generated for these parameters."""
        # If the pattern naturally only uses two colors, don't generate four-color version
        if self.uses_only_two_colors(params):
            return False
        
        # Check if this is a pattern type that has genuinely different behavior in four-color mode
        genuinely_different_patterns = [
            'alternating',     # Two-color: strict R/B alternation vs four-color: no consecutive same
            'balanced',        # Two-color: R=B vs four-color: R=B=Y=G
            'complex_specific_structure'  # R*B* vs R*B*Y*G*
        ]
        
        pattern_type = self.get_pattern_type()
        if hasattr(self, 'get_actual_pattern_type'):
            # For complex/numerical generators that handle multiple subtypes
            actual_pattern = self.get_actual_pattern_type(params)
            if actual_pattern in genuinely_different_patterns:
                return True
        elif pattern_type in genuinely_different_patterns:
            return True
        
        # For sequence-based patterns, check if sequence uses only R/B
        if 'sequence' in params:
            sequence = params['sequence']
            return any(char in sequence for char in ['Y', 'G'])
        
        # For character-based patterns, check if character is Y or G
        if 'target_char' in params:
            return params['target_char'] in ['Y', 'G']
        
        if 'forbidden_char' in params:
            return params['forbidden_char'] in ['Y', 'G']
        
        # Default: generate four-color version
        return True
    
    def select_parameters_by_mode(self, all_combinations: List[Any], count: int, override_mode: str = None) -> List[Any]:
        """Select parameters based on the configured generation mode (systematic vs random)."""
        max_capacity = len(all_combinations)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} problems, but can only generate {max_capacity} unique combinations.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        # Use override mode if provided, otherwise use global config
        mode = override_mode if override_mode is not None else GeneratorConfig.GENERATION_MODE
        
        if mode == 'random':
            # Random mode: shuffle the combinations and take the first count items
            shuffled_combinations = all_combinations.copy()
            random.shuffle(shuffled_combinations)
            return shuffled_combinations[:count]
        else:
            # Systematic mode (default): take combinations in order
            return all_combinations[:count] 