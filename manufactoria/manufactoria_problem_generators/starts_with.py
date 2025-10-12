"""
Starts With Pattern Generator

Generates problems that require strings to start with a specific sequence.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class StartsWithGenerator(BaseGenerator):
    """Generator for 'starts with sequence' patterns."""
    
    def get_pattern_type(self) -> str:
        return "starts_with"
    
    def get_difficulty(self) -> str:
        return "basic"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for starts_with patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        min_len, max_len = GeneratorConfig.get_sequence_length_range('starts_with')
        return sum(len(chars) ** length for length in range(min_len, max_len + 1))
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for starts_with patterns."""
        min_len, max_len = GeneratorConfig.get_sequence_length_range('starts_with')
        sequences = self.get_sequence_space(chars, min_len, max_len)
        max_capacity = len(sequences)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} starts_with problems, but can only generate {max_capacity} unique sequences.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        params_list = []
        
        for i in range(count):
            sequence = sequences[i]  # No more cycling
            params_list.append({'sequence': sequence})
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for starts_with."""
        cases = []
        sequence = pattern_info.get("sequence", "")
        
        if sequence:
            # Basic cases
            cases.append(sequence)  # Valid case
            cases.append(sequence + chars[0])  # Valid with extra
            # Invalid: doesn't start with sequence
            for char in chars:
                if char != sequence[0]:
                    cases.append(char + sequence)
                    break
            
            # Complex cases (>8 characters)
            # Valid: starts with sequence + long suffix
            suffix_len = GeneratorConfig.get_complex_case_length('medium_complex')
            long_suffix = ''.join(random.choice(chars) for _ in range(suffix_len))
            cases.append(sequence + long_suffix)
            
            # Valid: starts with sequence repeated and extended
            max_seq_len = GeneratorConfig.get_sequence_length_range('starts_with')[1]
            if len(sequence) <= max_seq_len:  # Avoid making it too long
                repeated_sequence = sequence * 3
                short_len, _ = GeneratorConfig.get_random_padding_length('short')
                extension = ''.join(random.choice(chars) for _ in range(short_len))
                cases.append(repeated_sequence + extension)
            
            # Invalid: almost starts with sequence but doesn't
            if len(sequence) > 1:
                # Wrong first character + sequence + long string
                wrong_first = [c for c in chars if c != sequence[0]][0]
                med_len, _ = GeneratorConfig.get_random_padding_length('medium')
                long_string = ''.join(random.choice(chars) for _ in range(med_len))
                cases.append(wrong_first + sequence[1:] + long_string)
            
            # Invalid: long string that doesn't start with sequence
            other_chars = [c for c in chars if c != sequence[0]]
            if other_chars:
                long_len = GeneratorConfig.get_complex_case_length('medium_complex')
                long_invalid = ''.join(random.choice(other_chars) for _ in range(long_len))
                cases.append(long_invalid)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a starts_with problem."""
        if params is None:
            # Generate random parameters
            min_len, max_len = GeneratorConfig.get_sequence_length_range('starts_with')
            seq_length = random.randint(min_len, max_len)
            sequence = ''.join(random.choice(chars) for _ in range(seq_length))
        else:
            sequence = params['sequence']
        
        def pattern_func(s: str) -> bool:
            return s.startswith(sequence)
        
        pattern_info = {"sequence": sequence}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        name = f"Starts with {sequence}"
        criteria = f"Accept if the tape starts with {sequence}; reject otherwise."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 