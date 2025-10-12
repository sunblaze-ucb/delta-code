"""
Numerical Operations Pattern Generator

Generates problems that perform numerical operations on binary strings:
1. Arithmetic operations: add, subtract
2. Bitwise operations: and, or, xor, not
3. Shift operations: floor division by powers of 2
"""

import random
import math
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class NumericalOperationsGenerator(BaseGenerator):
    """Generator for numerical operation patterns on binary strings."""
    
    @property
    def arithmetic_operations(self):
        """Get arithmetic operations from config."""
        return GeneratorConfig.ARITHMETIC_OPERATIONS
    
    @property
    def bitwise_operations(self):
        """Get bitwise operations from config."""
        return GeneratorConfig.BITWISE_OPERATIONS
    
    @property
    def shift_operations(self):
        """Get shift operations from config."""
        return GeneratorConfig.SHIFT_OPERATIONS
    
    @property
    def operand_values(self):
        """Get operand values from config."""
        return GeneratorConfig.OPERAND_VALUES
    
    def get_pattern_type(self) -> str:
        return "numerical_operations"
    
    def uses_only_two_colors(self, params: Dict[str, Any]) -> bool:
        """Numerical operations only use R (0) and B (1) for binary representation."""
        return True
    
    def get_actual_pattern_type(self, params: Dict[str, Any]) -> str:
        """Get the specific numerical operation pattern type."""
        operation = params.get('operation', 'add')
        operand = params.get('operand', 1)
        return f"numerical_operations_{operation}_{operand}"
    
    def get_difficulty(self) -> str:
        return "medium"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for numerical operations patterns."""
        all_operations = self.arithmetic_operations + self.bitwise_operations + self.shift_operations
        # For shift operations, we use powers of 2 as shift amounts (exponents)
        shift_powers = [i for i in range(1, GeneratorConfig.MAX_SHIFT_POWER + 1)]
        non_shift_operands = [x for x in self.operand_values if x <= 128]
        
        capacity = 0
        for operation in all_operations:
            if operation == 'floor_div':
                capacity += len(shift_powers)
            elif operation == 'not':
                capacity += 1  # NOT operation doesn't need operand
            else:
                capacity += len(non_shift_operands)
        
        return capacity
    
    def _generate_operation_parameters(self, count: int) -> List[Dict[str, Any]]:
        """Generate a list of operation and operand combinations."""
        all_operations = self.arithmetic_operations + self.bitwise_operations + self.shift_operations
        
        # Generate all combinations systematically
        all_combinations = []
        
        for operation in all_operations:
            if operation == 'floor_div':
                # For floor division, operand represents the power (x / 2^operand)
                for power in range(1, GeneratorConfig.MAX_SHIFT_POWER + 1):
                    all_combinations.append({
                        'operation': operation,
                        'operand': power,
                        'description': f'floor(x / 2^{power})'
                    })
            elif operation == 'not':
                # NOT operation doesn't need operand
                all_combinations.append({
                    'operation': operation,
                    'operand': None,
                    'description': 'bitwise NOT'
                })
            else:
                # Other operations use various operand values
                for operand in self.operand_values:
                    if operand <= 128:  # Keep operands reasonable
                        all_combinations.append({
                            'operation': operation,
                            'operand': operand,
                            'description': f'{operation} {operand}'
                        })
        
        # Calculate actual capacity
        max_capacity = len(all_combinations)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} numerical_operations problems, but can only generate {max_capacity} unique combinations.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        # Use base class method to select combinations based on generation mode
        result = self.select_parameters_by_mode(all_combinations, count)
        
        return result
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for numerical operations patterns."""
        return self._generate_operation_parameters(count)
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for numerical operations."""
        cases = []
        operation = pattern_info.get("operation", "add")
        operand = pattern_info.get("operand", 1)
        
        # Basic binary test cases
        cases.extend(["", "R", "B", "RB", "BR", "RRB", "BBR", "RBRB"])
        
        # Add test cases based on operation type
        if operation in self.arithmetic_operations:
            # For arithmetic operations, include values around the operand
            test_values = [0, 1, operand - 1, operand, operand + 1, operand * 2, operand * 4]
            
            # Add some larger values
            test_values.extend([64, 128, 255, 127, 63, 31, 15])
            
        elif operation in self.bitwise_operations:
            # For bitwise operations, include various bit patterns
            if operand is not None:
                test_values = [0, 1, operand, operand - 1, operand + 1]
                # Add values with interesting bit patterns
                test_values.extend([85, 170, 255, 127, 63, 31, 15])  # alternating and all-bits patterns
            else:
                # For NOT operation
                test_values = [0, 1, 15, 31, 63, 127, 255, 85, 170]
                
        elif operation == 'floor_div':
            # For floor division by 2^n, include values that showcase the effect
            divisor = 2 ** operand
            test_values = [0, 1, divisor - 1, divisor, divisor + 1, divisor * 2, divisor * 3]
            test_values.extend([255, 127, 63, 31, 15])
        
        # Convert test values to binary strings
        for val in test_values:
            if val >= 0 and val <= 1023:  # Keep values reasonable
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
    
    def _get_operation_function(self, operation: str, operand: int):
        """Return the appropriate operation function."""
        def operation_func(value: int) -> int:
            if operation == 'add':
                return value + operand
            elif operation == 'subtract':
                return value - operand  # Allow negative results
            elif operation == 'and':
                return value & operand
            elif operation == 'or':
                return value | operand
            elif operation == 'xor':
                return value ^ operand
            elif operation == 'not':
                # For NOT operation, we need to limit the result to reasonable bit width
                # Use 8-bit NOT for values up to 255
                bit_width = max(8, value.bit_length() if value > 0 else 1)
                mask = (1 << bit_width) - 1
                return (~value) & mask
            elif operation == 'floor_div':
                return value >> operand  # Right shift by operand (floor division by 2^operand)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        return operation_func
    
    def _get_operation_description(self, operation: str, operand: int) -> str:
        """Get human-readable description of the operation."""
        if operation == 'add':
            return f'add {operand}'
        elif operation == 'subtract':
            return f'subtract {operand}'
        elif operation == 'and':
            return f'bitwise AND with {operand}'
        elif operation == 'or':
            return f'bitwise OR with {operand}'
        elif operation == 'xor':
            return f'bitwise XOR with {operand}'
        elif operation == 'not':
            return 'bitwise NOT'
        elif operation == 'floor_div':
            return f'floor division by 2^{operand}'
        else:
            return f'{operation} {operand}'
    
    def _binary_to_int(self, s: str) -> int:
        """Convert binary string (R=0, B=1) to integer."""
        if not s:
            return 0
        value = 0
        for char in s:
            value = value * 2 + (1 if char == 'B' else 0)
        return value
    
    def _int_to_binary(self, value: int) -> str:
        """Convert integer to binary string (R=0, B=1).
        
        Rules:
        - If value is 0 or negative, return empty string
        - If value > 0, return binary representation where result starts with 'B'
        """
        if value <= 0:
            return ""  # Empty string represents 0 or negative values
        binary_str = bin(value)[2:]  # Remove '0b' prefix
        result = ''.join('B' if bit == '1' else 'R' for bit in binary_str)
        # Ensure positive results start with 'B' (they should by definition since MSB=1)
        return result
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a numerical operations problem."""
        if params is None:
            # Generate random parameters
            all_operations = self.arithmetic_operations + self.bitwise_operations + self.shift_operations
            operation = random.choice(all_operations)
            
            if operation == 'floor_div':
                operand = random.randint(1, GeneratorConfig.MAX_SHIFT_POWER)  # Power for 2^operand
            elif operation == 'not':
                operand = None
            else:
                operand = random.choice(self.operand_values)
        else:
            operation = params['operation']
            operand = params.get('operand')
        
        pattern_info = {"operation": operation, "operand": operand}
        
        operation_func = self._get_operation_function(operation, operand)
        
        def pattern_func(s: str) -> bool:
            """For numerical operations, we accept all input strings."""
            return True
        
        def expected_output_func(s: str) -> str:
            """Return the expected output after applying the operation."""
            input_value = self._binary_to_int(s)
            result_value = operation_func(input_value)
            return self._int_to_binary(result_value)
        
        # Create test cases with output checking enabled
        test_cases = []
        seen_inputs = set()
        
        # Generate deterministic test cases
        deterministic_cases = self._generate_deterministic_test_cases('numerical_operations', chars, pattern_info)
        
        # Add deterministic cases with expected outputs
        for test_input in deterministic_cases:
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_output = expected_output_func(test_input)
                test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                    "expected_accepted": True,  # All inputs should be accepted
                    "check_output": True,  # We check the output
                    "description": f"Input '{test_input}' should output '{expected_output}'"
                })
        
        # Add more test cases to reach minimum requirements
        while len(test_cases) < GeneratorConfig.MIN_ACCEPTING_CASES + GeneratorConfig.MIN_REJECTING_CASES:
            test_input = self.generate_test_string(chars)
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_output = expected_output_func(test_input)
                test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                    "expected_accepted": True,
                    "check_output": True,
                    "description": f"Input '{test_input}' should output '{expected_output}'"
                })
        
        # Limit to max test cases
        test_cases = test_cases[:GeneratorConfig.MAX_TEST_CASES]
        
        operation_desc = self._get_operation_description(operation, operand)
        problem_id = str(__import__('uuid').uuid4())
        name = f"Numerical {operation_desc.title()}"
        criteria = f"Treat Blue as 1 and Red as 0. Apply {operation_desc} to the binary number and output the result. If the result is zero or negative, output an empty tape. If the result is positive, output the binary representation (which will start with B)."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color,
            description=f"Convert input to binary number, apply {operation_desc}, and output the result as binary. Zero or negative results become empty tape, positive results start with B."
        )