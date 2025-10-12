"""
Append Sequence Pattern Generator

Generates problems that require appending a specific sequence to the end of the input tape.
Unlike other generators, this checks both input pattern acceptance AND output correctness.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class AppendSequenceGenerator(BaseGenerator):
    """Generator for 'append sequence' patterns."""
    
    def get_pattern_type(self) -> str:
        return "append_sequence"
    
    def get_difficulty(self) -> str:
        return "basic"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for append_sequence patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        min_len, max_len = GeneratorConfig.get_sequence_length_range('append_sequence')
        return sum(len(chars) ** length for length in range(min_len, max_len + 1))
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for append_sequence patterns."""
        min_len, max_len = GeneratorConfig.get_sequence_length_range('append_sequence')
        sequences = self.get_sequence_space(chars, min_len, max_len)
        max_capacity = len(sequences)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} append_sequence problems, but can only generate {max_capacity} unique sequences.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        params_list = []
        
        for i in range(count):
            sequence = sequences[i]  # No more cycling
            params_list.append({'sequence': sequence})
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for append_sequence."""
        cases = []
        sequence = pattern_info.get("sequence", "")
        
        if sequence:
            # Basic cases - these should be accepted by the pattern function
            # and will have the sequence appended
            cases.append("")  # Empty input should be accepted
            cases.append(chars[0])  # Single character input
            cases.append(chars[0] + chars[1] if len(chars) > 1 else chars[0])  # Two character input
            
            # Medium length inputs that should be accepted
            medium_input = ''.join(random.choice(chars) for _ in range(4))
            cases.append(medium_input)
            
            # Longer inputs that should be accepted
            long_input = ''.join(random.choice(chars) for _ in range(8))
            cases.append(long_input)
            
            # Complex cases (>8 characters) that should be accepted
            complex_len = GeneratorConfig.get_complex_case_length('medium_complex')
            complex_input = ''.join(random.choice(chars) for _ in range(complex_len))
            cases.append(complex_input)
            
            # Very long input
            very_long_len = GeneratorConfig.get_complex_case_length('long_complex')
            very_long_input = ''.join(random.choice(chars) for _ in range(very_long_len))
            cases.append(very_long_input)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate an append_sequence problem."""
        if params is None:
            # Generate random parameters
            min_len, max_len = GeneratorConfig.get_sequence_length_range('append_sequence')
            seq_length = random.randint(min_len, max_len)
            sequence = ''.join(random.choice(chars) for _ in range(seq_length))
        else:
            sequence = params['sequence']
        
        def pattern_func(s: str) -> bool:
            """For append_sequence, we accept ALL valid input strings."""
            # Accept all strings (the machine should append the sequence to any input)
            return True
        
        def expected_output_func(s: str) -> str:
            """Return the expected output: input + appended sequence."""
            return s + sequence
        
        pattern_info = {"sequence": sequence}
        
        # Create test cases with output checking enabled
        test_cases = []
        seen_inputs = set()
        
        # Generate deterministic test cases
        deterministic_cases = self._generate_deterministic_test_cases('append_sequence', chars, pattern_info)
        
        # Add deterministic cases with expected outputs
        for test_input in deterministic_cases:
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_output = expected_output_func(test_input)
                test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                    "expected_accepted": True,  # All inputs should be accepted
                    "check_output": True,  # KEY DIFFERENCE: We check the output
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
        
        problem_id = str(__import__('uuid').uuid4())
        name = f"Append {sequence}"
        criteria = f"Accept any input and append the sequence '{sequence}' to the end of the tape."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color,
            description=f"The machine should read any input sequence and append '{sequence}' to the end."
        )