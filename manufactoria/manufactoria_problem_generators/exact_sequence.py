"""
Exact Sequence Pattern Generator

Generates problems that require strings to exactly match a specific sequence.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class ExactSequenceGenerator(BaseGenerator):
    """Generator for 'exactly equals sequence' patterns."""
    
    def get_pattern_type(self) -> str:
        return "exact_sequence"
    
    def get_difficulty(self) -> str:
        return "basic"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for exact_sequence patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        min_len, max_len = GeneratorConfig.get_sequence_length_range('exact_sequence')
        return sum(len(chars) ** length for length in range(min_len, max_len + 1))
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for exact_sequence patterns."""
        min_len, max_len = GeneratorConfig.get_sequence_length_range('exact_sequence')
        sequences = self.get_sequence_space(chars, min_len, max_len)
        max_capacity = len(sequences)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} exact_sequence problems, but can only generate {max_capacity} unique sequences.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        params_list = []
        
        for i in range(count):
            sequence = sequences[i]  # No more cycling
            params_list.append({'sequence': sequence})
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for exact_sequence."""
        cases = []
        sequence = pattern_info.get("sequence", "")
        
        if sequence:
            # Basic cases
            cases.append(sequence)  # Valid case
            if len(sequence) > 1:
                cases.append(sequence[:-1])  # Remove last char
                cases.append(sequence + chars[0])  # Add extra char
            if len(sequence) > 0:
                # Change one character
                for i, char in enumerate(chars):
                    if char != sequence[0]:
                        cases.append(char + sequence[1:] if len(sequence) > 1 else char)
                        break
            
            # Complex cases (>8 characters)
            # Invalid: sequence + long suffix (not exact match)
            long_len, _ = GeneratorConfig.get_random_padding_length('long')
            long_suffix = ''.join(random.choice(chars) for _ in range(long_len))
            cases.append(sequence + long_suffix)
            
            # Invalid: long prefix + sequence (not exact match)
            medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
            long_prefix = ''.join(random.choice(chars) for _ in range(medium_len))
            cases.append(long_prefix + sequence)
            
            # Invalid: sequence embedded in long string (not exact match)
            short_len, _ = GeneratorConfig.get_random_padding_length('short')
            medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
            prefix = ''.join(random.choice(chars) for _ in range(short_len))
            suffix = ''.join(random.choice(chars) for _ in range(medium_len))
            cases.append(prefix + sequence + suffix)
            
            # Invalid: similar long sequence but with one character different
            if len(sequence) >= 3:
                modified_seq = list(sequence)
                # Change middle character
                mid_idx = len(modified_seq) // 2
                for char in chars:
                    if char != modified_seq[mid_idx]:
                        modified_seq[mid_idx] = char
                        break
                medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
                long_similar = ''.join(modified_seq) + ''.join(random.choice(chars) for _ in range(medium_len))
                cases.append(long_similar)
            
            # Invalid: completely different long string
            complex_len = GeneratorConfig.get_complex_case_length('medium_complex')
            different_long = ''.join(random.choice(chars) for _ in range(complex_len))
            while different_long == sequence:  # Ensure it's different
                different_long = ''.join(random.choice(chars) for _ in range(complex_len))
            cases.append(different_long)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate an exact_sequence problem."""
        if params is None:
            # Generate random parameters
            min_len, max_len = GeneratorConfig.get_sequence_length_range('exact_sequence')
            seq_length = random.randint(min_len, max_len)
            sequence = ''.join(random.choice(chars) for _ in range(seq_length))
        else:
            sequence = params['sequence']
        
        def pattern_func(s: str) -> bool:
            return s == sequence
        
        pattern_info = {"sequence": sequence}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        name = f"Exactly {sequence}"
        criteria = f"Accept if the tape is exactly {sequence}; reject otherwise."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 