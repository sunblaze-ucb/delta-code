"""
Numerical Max/Min Operations Pattern Generator

Generates problems that perform max/min operations between the input binary number and a constant:
1. max(n, input) - returns the maximum of n and the input value
2. min(n, input) - returns the minimum of n and the input value
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class NumericalMaxMinGenerator(BaseGenerator):
    """Generator for numerical max/min operation patterns on binary strings."""
    
    @property
    def max_min_operations(self):
        """Get max/min operations from config."""
        return GeneratorConfig.MAX_MIN_OPERATIONS
    
    @property
    def operand_range(self):
        """Get operand range from config."""
        return GeneratorConfig.MAX_MIN_OPERAND_RANGE
    
    def get_pattern_type(self) -> str:
        return "numerical_max_min"
    
    def uses_only_two_colors(self, params: Dict[str, Any]) -> bool:
        """Max/min operations only use R (0) and B (1) for binary representation."""
        return True
    
    def get_actual_pattern_type(self, params: Dict[str, Any]) -> str:
        """Get the specific max/min operation pattern type."""
        operation = params.get('operation', 'max')
        operand = params.get('operand', 1)
        return f"numerical_max_min_{operation}_{operand}"
    
    def get_difficulty(self) -> str:
        return "easy"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for numerical max/min patterns."""
        # Two operations (max, min) times the range of operand values (1 to 256)
        min_operand, max_operand = self.operand_range
        operand_count = max_operand - min_operand + 1  # +1 because range is inclusive
        return len(self.max_min_operations) * operand_count
    
    def _generate_max_min_parameters(self, count: int) -> List[Dict[str, Any]]:
        """Generate a list of max/min operation and operand combinations."""
        
        # Generate all combinations systematically
        all_combinations = []
        min_operand, max_operand = self.operand_range
        
        for operation in self.max_min_operations:
            for operand in range(min_operand, max_operand + 1):  # +1 because range is exclusive at end
                all_combinations.append({
                    'operation': operation,
                    'operand': operand,
                    'description': f'{operation}({operand}, input)'
                })
        
        # Calculate actual capacity
        max_capacity = len(all_combinations)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} numerical_max_min problems, but can only generate {max_capacity} unique combinations.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        # Use base class method to select combinations based on generation mode
        result = self.select_parameters_by_mode(all_combinations, count)
        
        return result
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for numerical max/min patterns."""
        return self._generate_max_min_parameters(count)
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for numerical max/min operations."""
        cases = []
        operation = pattern_info.get("operation", "max")
        operand = pattern_info.get("operand", 1)
        
        # Basic binary test cases
        cases.extend(["", "R", "B", "RB", "BR", "RRB", "BBR", "RBRB"])
        
        # Add test cases based on operation type and operand value
        # Include values below, at, and above the operand to test the comparison logic
        test_values = [0, 1]
        
        # Add values specifically around the operand
        if operand > 2:
            test_values.extend([operand - 2, operand - 1])
        test_values.extend([operand, operand + 1, operand + 2])
        
        # Add some powers of 2 and interesting values
        powers_of_2 = [2, 4, 8, 16, 32, 64, 128]
        for power in powers_of_2:
            if power != operand:  # Don't duplicate the operand
                test_values.append(power)
        
        # Add some values with interesting bit patterns
        interesting_values = [3, 5, 7, 15, 31, 63, 127, 85, 170, 255]
        test_values.extend(interesting_values)
        
        # Convert test values to binary strings, filtering duplicates and keeping reasonable size
        seen_values = set()
        for val in test_values:
            if val >= 0 and val <= 1023 and val not in seen_values:  # Keep values reasonable
                seen_values.add(val)
                if val == 0:
                    cases.append("")  # Empty string represents 0
                else:
                    binary_str = bin(val)[2:]  # Remove '0b' prefix
                    candidate = ''.join('B' if bit == '1' else 'R' for bit in binary_str)
                    cases.append(candidate)
        
        # Add some longer sequences for complex cases
        complex_patterns = [
            "BRRRRRR",       # 64
            "BRRRRRRR",      # 128
            "BRRRRRRRR",     # 256
            "BRBRBRBR",      # 170
            "RBBRBRRBR",     # 356
            "BRBBBBBB",      # 191
            "RBRBRBRR",      # 90
            "BRBRBR",        # 85
            "BBRBBRBR",      # 219
            "RBBRRBBR"       # 155
        ]
        cases.extend(complex_patterns)
        
        return cases
    
    def _get_max_min_function(self, operation: str, operand: int):
        """Return the appropriate max/min operation function."""
        def operation_func(value: int) -> int:
            if operation == 'max':
                return max(operand, value)
            elif operation == 'min':
                return min(operand, value)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        return operation_func
    
    def _get_operation_description(self, operation: str, operand: int) -> str:
        """Get human-readable description of the operation."""
        if operation == 'max':
            return f'maximum of {operand} and input'
        elif operation == 'min':
            return f'minimum of {operand} and input'
        else:
            return f'{operation}({operand}, input)'
    
    def _binary_to_int(self, s: str) -> int:
        """Convert binary string (R=0, B=1) to integer."""
        if not s:
            return 0
        value = 0
        for char in s:
            value = value * 2 + (1 if char == 'B' else 0)
        return value
    
    def _int_to_binary(self, value: int) -> str:
        """Convert integer to binary string (R=0, B=1)."""
        if value == 0:
            return ""  # Empty string represents 0
        binary_str = bin(value)[2:]  # Remove '0b' prefix
        return ''.join('B' if bit == '1' else 'R' for bit in binary_str)
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a numerical max/min problem."""
        if params is None:
            # Generate random parameters
            operation = random.choice(self.max_min_operations)
            min_operand, max_operand = self.operand_range
            operand = random.randint(min_operand, max_operand)
        else:
            operation = params['operation']
            operand = params.get('operand', 1)
        
        pattern_info = {"operation": operation, "operand": operand}
        
        max_min_func = self._get_max_min_function(operation, operand)
        
        def pattern_func(s: str) -> bool:
            """For numerical max/min operations, we accept all input strings."""
            return True
        
        def expected_output_func(s: str) -> str:
            """Return the expected output after applying the max/min operation."""
            input_value = self._binary_to_int(s)
            result_value = max_min_func(input_value)
            return self._int_to_binary(result_value)
        
        # Create test cases with output checking enabled
        test_cases = []
        seen_inputs = set()
        
        # Generate deterministic test cases
        deterministic_cases = self._generate_deterministic_test_cases('numerical_max_min', chars, pattern_info)
        
        # Add deterministic cases with expected outputs
        for test_input in deterministic_cases:
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_output = expected_output_func(test_input)
                input_val = self._binary_to_int(test_input)
                result_val = self._binary_to_int(expected_output)
                test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                    "expected_accepted": True,  # All inputs should be accepted
                    "check_output": True,  # We check the output
                    "description": f"Input '{test_input}' (value {input_val}) should output '{expected_output}' (value {result_val})"
                })
        
        # Add more test cases to reach minimum requirements
        while len(test_cases) < GeneratorConfig.MIN_ACCEPTING_CASES + GeneratorConfig.MIN_REJECTING_CASES:
            test_input = self.generate_test_string(chars)
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_output = expected_output_func(test_input)
                input_val = self._binary_to_int(test_input)
                result_val = self._binary_to_int(expected_output)
                test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                    "expected_accepted": True,
                    "check_output": True,
                    "description": f"Input '{test_input}' (value {input_val}) should output '{expected_output}' (value {result_val})"
                })
        
        # Limit to max test cases
        test_cases = test_cases[:GeneratorConfig.MAX_TEST_CASES]
        
        operation_desc = self._get_operation_description(operation, operand)
        problem_id = str(__import__('uuid').uuid4())
        name = f"Numerical {operation.upper()}({operand}, input)"
        criteria = f"Treat Blue as 1 and Red as 0. Output the {operation_desc}."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color,
            description=f"Convert input to binary number, compute {operation_desc}, and output the result as binary."
        )