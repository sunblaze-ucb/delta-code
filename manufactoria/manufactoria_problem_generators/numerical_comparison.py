"""
Numerical Comparison Pattern Generator

Generates problems that treat the tape as binary numbers and check various comparisons against a threshold.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class NumericalComparisonGenerator(BaseGenerator):
    """Generator for numerical comparison patterns (binary operations)."""
    
    # Define the supported comparison operators
    COMPARISON_OPERATORS = ['>', '>=', '<', '<=', '=']
    
    def get_pattern_type(self) -> str:
        return "numerical_comparison"
    
    def uses_only_two_colors(self, params: Dict[str, Any]) -> bool:
        """Numerical patterns only use R (0) and B (1) for binary operations."""
        return True
    
    def get_actual_pattern_type(self, params: Dict[str, Any]) -> str:
        """Get the specific numerical pattern type."""
        threshold = params.get('threshold', 0)
        operator = params.get('operator', '>')
        return f"numerical_comparison_{operator}_{threshold}"
    
    def get_difficulty(self) -> str:
        return "hard"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for numerical comparison patterns."""
        # Can generate problems for all thresholds from min to max threshold for each operator
        min_threshold, max_threshold = GeneratorConfig.NUMERICAL_THRESHOLDS
        num_operators = len(self.COMPARISON_OPERATORS)
        threshold_range = max_threshold - min_threshold + 1
        return threshold_range * num_operators
    
    def _generate_comparison_parameters(self, count: int) -> List[Dict[str, Any]]:
        """Generate a list of threshold and operator combinations for comparison problems."""
        min_threshold, max_threshold = GeneratorConfig.NUMERICAL_THRESHOLDS
        operators = self.COMPARISON_OPERATORS
        threshold_range = max_threshold - min_threshold + 1
        
        # Calculate actual capacity
        max_capacity = len(operators) * threshold_range
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} numerical_comparison problems, but can only generate {max_capacity} unique combinations.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        # Generate all combinations systematically first
        all_combinations = []
        for operator in operators:
            for threshold in range(min_threshold, max_threshold + 1):
                all_combinations.append({
                    'threshold': threshold,
                    'operator': operator
                })
        
        # Use base class method to select combinations based on generation mode
        result = self.select_parameters_by_mode(all_combinations, count)
        
        return result
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for numerical comparison patterns."""
        return self._generate_comparison_parameters(count)
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for numerical comparison patterns."""
        cases = []
        threshold = pattern_info.get("threshold", 0)
        operator = pattern_info.get("operator", '>')
        
        # Basic binary test cases
        cases.extend(["R", "B", "RB", "BR", "RRB", "BBR", "RBRB"])
        
        if threshold > 0:
            # Generate binary representations around the threshold
            test_values = []
            
            # Always include the threshold itself
            test_values.append(threshold)
            
            # Add values around the threshold based on operator
            if operator in ['>', '>=']:
                # Add values below, at, and above threshold
                test_values.extend([threshold - 2, threshold - 1, threshold + 1, threshold + 2, threshold * 2])
            elif operator in ['<', '<=']:
                # Add values below, at, and above threshold
                test_values.extend([max(0, threshold - 2), max(0, threshold - 1), threshold + 1, threshold + 2])
            elif operator == '=':
                # For equality, focus on values around the threshold
                test_values.extend([max(0, threshold - 1), threshold + 1, threshold - 2, threshold + 2])
            
            # Convert to binary strings
            for val in test_values:
                if val >= 0:
                    binary_str = bin(val)[2:]  # Remove '0b' prefix
                    candidate = ''.join('B' if bit == '1' else 'R' for bit in binary_str)
                    cases.append(candidate)
            
            # Additional test cases based on operator type
            if operator in ['>', '>=']:
                # Large numbers that are definitely greater than threshold
                large_numbers = [threshold * 4, threshold + 100, threshold + 50]
                for num in large_numbers:
                    if num < GeneratorConfig.MAX_BINARY_STRING_LENGTH:
                        binary_str = bin(num)[2:]
                        candidate = ''.join('B' if bit == '1' else 'R' for bit in binary_str)
                        cases.append(candidate)
                        
            elif operator in ['<', '<=']:
                # Small numbers that are definitely less than threshold
                if threshold > 10:
                    small_numbers = [threshold // 2, threshold // 4, max(1, threshold - 10), max(1, threshold - 5)]
                    for num in small_numbers:
                        binary_str = bin(num)[2:]
                        candidate = ''.join('B' if bit == '1' else 'R' for bit in binary_str)
                        cases.append(candidate)
            
            # Edge cases with specific patterns for all operators
            if threshold >= 8:
                # Some longer sequences
                cases.extend([
                    "BRRRRRR",       # 64
                    "BRRRRRRR",      # 128
                    "BRRRRRRRR",     # 256
                    "BRBRBRBR",      # 170
                    "RBBRBRRBR",     # 356
                    "BRBBBBBB",      # 191
                    "RBRBRBRR",      # 90
                    "BRBRBR"         # 85
                ])
        
        return cases
    
    def _get_comparison_function(self, operator: str, threshold: int):
        """Return the appropriate comparison function based on operator."""
        def compare_func(value: int) -> bool:
            if operator == '>':
                return value > threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<':
                return value < threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '=':
                return value == threshold
            else:
                raise ValueError(f"Unknown operator: {operator}")
        
        return compare_func
    
    def _get_operator_description(self, operator: str) -> str:
        """Get human-readable description of the operator."""
        descriptions = {
            '>': 'greater than',
            '>=': 'greater than or equal to',
            '<': 'less than',
            '<=': 'less than or equal to',
            '=': 'equal to'
        }
        return descriptions.get(operator, operator)
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a numerical comparison problem."""
        if params is None:
            # Generate random threshold and operator
            min_threshold, max_threshold = GeneratorConfig.NUMERICAL_THRESHOLDS
            threshold = random.randint(min_threshold, max_threshold)
            operator = random.choice(self.COMPARISON_OPERATORS)
        else:
            threshold = params['threshold']
            operator = params.get('operator', '>')
        
        pattern_info = {"threshold": threshold, "operator": operator}
        
        comparison_func = self._get_comparison_function(operator, threshold)
        
        def pattern_func(s: str) -> bool:
            if not s:
                # Empty string represents 0
                return comparison_func(0)
            value = 0
            for char in s:
                value = value * 2 + (1 if char == 'B' else 0)
            return comparison_func(value)
        
        operator_desc = self._get_operator_description(operator)
        criteria_text = f"Treat Blue as 1 and Red as 0. Accept if the binary number is {operator_desc} {threshold}; reject otherwise."
        name = f"Binary {operator_desc.title()} {threshold}"
        
        test_cases = self.create_test_cases(
            pattern_func, chars, False, f"numerical_comparison", pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        
        return self.create_problem_dict(
            problem_id, name, criteria_text, test_cases, 
            self.get_difficulty(), f"numerical_comparison", is_four_color
        ) 