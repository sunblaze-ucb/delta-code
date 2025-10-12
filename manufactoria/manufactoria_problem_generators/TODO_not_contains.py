"""
Not Contains Pattern Generator

Generates problems that require strings to not contain a specific character.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class NotContainsGenerator(BaseGenerator):
    """Generator for 'does not contain' patterns."""
    
    def get_pattern_type(self) -> str:
        return "not_contains"
    
    def get_difficulty(self) -> str:
        return "easy"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for not_contains patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        return len(chars)
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for not_contains patterns."""
        all_chars = chars.copy()
        params_list = []
        
        for i in range(count):
            char = all_chars[i % len(all_chars)]
            params_list.append({'forbidden_char': char})
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for not_contains."""
        cases = []
        forbidden_char = pattern_info.get("forbidden_char", chars[0])
        other_chars = [c for c in chars if c != forbidden_char]
        
        # Basic cases
        if other_chars:
            # Valid cases
            cases.append(other_chars[0])
            cases.append(other_chars[0] * 2)
        # Invalid case
        cases.append(forbidden_char)
        
        # Complex cases (>8 characters)
        if other_chars:
            # Valid: long string with only allowed characters
            complex_len = GeneratorConfig.get_complex_case_length('medium_complex')
            long_valid = ''.join(random.choice(other_chars) for _ in range(complex_len))
            cases.append(long_valid)
            
            # Valid: very long string with mixed allowed characters
            if len(other_chars) > 1:
                long_complex_len = GeneratorConfig.get_complex_case_length('long_complex')
                long_mixed = ''.join(random.choice(other_chars) for _ in range(long_complex_len))
                cases.append(long_mixed)
            
            # Valid: pattern-like string with only allowed characters
            medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
            pattern_valid = other_chars[0] * 4 + (other_chars[1] if len(other_chars) > 1 else other_chars[0]) * medium_len
            cases.append(pattern_valid)
        
        # Invalid: forbidden char at beginning of long string
        if other_chars:
            long_len, _ = GeneratorConfig.get_random_padding_length('long')
            long_suffix = ''.join(random.choice(other_chars) for _ in range(long_len))
            cases.append(forbidden_char + long_suffix)
        
        # Invalid: forbidden char at end of long string
        if other_chars:
            long_len, _ = GeneratorConfig.get_random_padding_length('long')
            long_prefix = ''.join(random.choice(other_chars) for _ in range(long_len - 1))
            cases.append(long_prefix + forbidden_char)
        
        # Invalid: forbidden char in middle of long string
        if other_chars:
            short_len, _ = GeneratorConfig.get_random_padding_length('short')
            medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
            prefix = ''.join(random.choice(other_chars) for _ in range(short_len))
            suffix = ''.join(random.choice(other_chars) for _ in range(medium_len))
            cases.append(prefix + forbidden_char + suffix)
        
        # Invalid: multiple forbidden chars in long string
        if other_chars:
            complex_len = GeneratorConfig.get_complex_case_length('medium_complex')
            long_string = []
            forbidden_positions = random.sample(range(complex_len), min(3, complex_len))  # 3 forbidden chars
            for i in range(complex_len):
                if i in forbidden_positions:
                    long_string.append(forbidden_char)
                else:
                    long_string.append(random.choice(other_chars))
            cases.append(''.join(long_string))
        
        # Invalid: only forbidden chars
        long_len, _ = GeneratorConfig.get_random_padding_length('long')
        cases.append(forbidden_char * long_len)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a not_contains problem."""
        if params is None:
            # Generate random parameters
            forbidden_char = random.choice(chars)
        else:
            forbidden_char = params['forbidden_char']
        
        def pattern_func(s: str) -> bool:
            return forbidden_char not in s
        
        pattern_info = {"forbidden_char": forbidden_char}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        name = f"No {forbidden_char}"
        criteria = f"Accept if the tape contains no {forbidden_char.lower()}; reject otherwise."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 