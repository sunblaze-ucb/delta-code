"""
Ends With Pattern Generator

Generates problems that require strings to end with a specific sequence.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class EndsWithGenerator(BaseGenerator):
    """Generator for 'ends with sequence' patterns."""
    
    def get_pattern_type(self) -> str:
        return "ends_with"
    
    def get_difficulty(self) -> str:
        return "medium"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for ends_with patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        min_len, max_len = GeneratorConfig.get_sequence_length_range('ends_with')
        return sum(len(chars) ** length for length in range(min_len, max_len + 1))
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for ends_with patterns."""
        min_len, max_len = GeneratorConfig.get_sequence_length_range('ends_with')
        sequences = self.get_sequence_space(chars, min_len, max_len)
        max_capacity = len(sequences)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} ends_with problems, but can only generate {max_capacity} unique sequences.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        params_list = []
        
        for i in range(count):
            sequence = sequences[i]  # No more cycling
            params_list.append({'sequence': sequence})
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for ends_with."""
        cases = []
        sequence = pattern_info.get("sequence", "")
        
        if sequence:
            # Basic cases
            cases.append(sequence)  # Valid case
            cases.append(chars[0] + sequence)  # Valid with prefix
            # Invalid: doesn't end with sequence
            cases.append(sequence + chars[0])
            
            # Complex cases (>8 characters)
            # Valid: long prefix + sequence
            long_len, _ = GeneratorConfig.get_random_padding_length('long')
            long_prefix = ''.join(random.choice(chars) for _ in range(long_len))
            cases.append(long_prefix + sequence)
            
            # Valid: very long prefix + sequence
            very_long_len = GeneratorConfig.get_complex_case_length('long_complex')
            very_long_prefix = ''.join(random.choice(chars) for _ in range(very_long_len))
            cases.append(very_long_prefix + sequence)
            
            # Valid: pattern-like prefix + sequence
            max_seq_len = GeneratorConfig.get_sequence_length_range('ends_with')[1]
            if len(sequence) <= max_seq_len:  # Avoid making it too long
                short_len, _ = GeneratorConfig.get_random_padding_length('short')
                pattern_prefix = sequence * 3 + ''.join(random.choice(chars) for _ in range(short_len))
                cases.append(pattern_prefix + sequence)
            
            # Invalid: sequence + long suffix (doesn't end with sequence)
            medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
            long_suffix = ''.join(random.choice(chars) for _ in range(medium_len))
            cases.append(sequence + long_suffix)
            
            # Invalid: long string with sequence in middle
            short_len, _ = GeneratorConfig.get_random_padding_length('short')
            medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
            prefix = ''.join(random.choice(chars) for _ in range(short_len))
            suffix = ''.join(random.choice(chars) for _ in range(medium_len))
            cases.append(prefix + sequence + suffix)
            
            # Invalid: almost ends with sequence but missing last char
            if len(sequence) > 1:
                almost_sequence = sequence[:-1]
                medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
                long_prefix2 = ''.join(random.choice(chars) for _ in range(medium_len))
                cases.append(long_prefix2 + almost_sequence)
            
            # Invalid: ends with similar sequence but one char different
            if len(sequence) >= 2:
                modified_seq = list(sequence)
                # Change last character
                for char in chars:
                    if char != modified_seq[-1]:
                        modified_seq[-1] = char
                        break
                medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
                long_prefix3 = ''.join(random.choice(chars) for _ in range(medium_len - 1))
                cases.append(long_prefix3 + ''.join(modified_seq))
            
            # Invalid: completely different long string
            different_chars = [c for c in chars if c not in sequence]
            if different_chars:
                complex_len = GeneratorConfig.get_complex_case_length('medium_complex')
                long_different = ''.join(random.choice(different_chars) for _ in range(complex_len))
                cases.append(long_different)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate an ends_with problem."""
        if params is None:
            # Generate random parameters
            min_len, max_len = GeneratorConfig.get_sequence_length_range('ends_with')
            seq_length = random.randint(min_len, max_len)
            sequence = ''.join(random.choice(chars) for _ in range(seq_length))
        else:
            sequence = params['sequence']
        
        def pattern_func(s: str) -> bool:
            return s.endswith(sequence)
        
        pattern_info = {"sequence": sequence}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        name = f"Ends with {sequence}"
        criteria = f"Accept if the tape ends with {sequence}; reject otherwise."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 