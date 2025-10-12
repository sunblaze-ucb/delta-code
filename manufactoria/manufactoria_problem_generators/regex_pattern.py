"""
Regex Pattern Generator

Generates problems that require strings to match regex patterns built from concatenated
sub-patterns with repetition operators (+, ?, *).
"""

import random
import re
from typing import List, Dict, Any, Tuple
from itertools import product
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class RegexPatternGenerator(BaseGenerator):
    """Generator for regex patterns with concatenation and repetition operators."""
    
    def __init__(self):
        super().__init__()
        # Don't cache config values - read them dynamically to support difficulty overrides
    
    def get_pattern_type(self) -> str:
        return "regex_pattern"
    
    def get_difficulty(self) -> str:
        return "hard"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for regex patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        # Support both single value and range for REGEX_MAX_PATTERN_LENGTH
        if isinstance(GeneratorConfig.REGEX_MAX_PATTERN_LENGTH, (list, tuple)):
            min_len, max_len = GeneratorConfig.REGEX_MAX_PATTERN_LENGTH
        else:
            min_len, max_len = 1, GeneratorConfig.REGEX_MAX_PATTERN_LENGTH
        
        # Calculate all possible base patterns within the length range
        base_patterns = sum(len(chars) ** length for length in range(min_len, max_len + 1))
        operators = len(GeneratorConfig.REGEX_OPERATORS)
        # Each of the concatenation_count positions can independently be any of (base_patterns * operators) combinations
        return (base_patterns * operators) ** GeneratorConfig.REGEX_CONCATENATION_COUNT
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for regex patterns."""
        concat_count = GeneratorConfig.REGEX_CONCATENATION_COUNT
        # Support both single value and range for REGEX_MAX_PATTERN_LENGTH
        if isinstance(GeneratorConfig.REGEX_MAX_PATTERN_LENGTH, (list, tuple)):
            min_pattern_len, max_pattern_len = GeneratorConfig.REGEX_MAX_PATTERN_LENGTH
        else:
            min_pattern_len, max_pattern_len = 1, GeneratorConfig.REGEX_MAX_PATTERN_LENGTH
        operators = GeneratorConfig.REGEX_OPERATORS
        
        # Generate all possible base patterns within the length range
        base_patterns = []
        for length in range(min_pattern_len, max_pattern_len + 1):
            for pattern in product(chars, repeat=length):
                base_patterns.append(''.join(pattern))
        
        # Generate all possible combinations of (base_pattern, operator) for each concatenation position
        pattern_operator_combos = list(product(base_patterns, operators))
        
        # Generate all possible complete regex patterns (concatenation_count positions)
        all_combinations = list(product(pattern_operator_combos, repeat=concat_count))
        
        # Use base class method to select combinations based on generation mode
        selected_combinations = self.select_parameters_by_mode(all_combinations, count)
        
        params_list = []
        
        for combination in selected_combinations:
            pattern_parts = list(combination)  # This is a list of (base_pattern, operator) tuples
            
            params_list.append({
                'pattern_parts': pattern_parts,
                'concatenation_count': concat_count,
                'min_pattern_length': min_pattern_len,
                'max_pattern_length': max_pattern_len
            })
        
        return params_list
    
    def _generate_base_pattern(self, chars: List[str], max_length: int) -> str:
        """Generate a base pattern from available characters (legacy method, no longer used)."""
        length = random.randint(1, max_length)
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _build_regex_pattern(self, pattern_parts: List[Tuple[str, str]]) -> str:
        """Build a regex pattern from pattern parts and operators."""
        regex_parts = []
        for base_pattern, operator in pattern_parts:
            if operator == '+':
                regex_parts.append(f"({base_pattern})+")
            elif operator == '?':
                regex_parts.append(f"({base_pattern})?")
            elif operator == '*':
                regex_parts.append(f"({base_pattern})*")
        
        return ''.join(regex_parts)
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for regex patterns."""
        cases = []
        pattern_parts = pattern_info.get("pattern_parts", [])
        
        if not pattern_parts:
            return cases
        
        # ===== VALID CASES =====
        
        # 1. Minimal valid case (minimal repetitions)
        minimal_case = ""
        for base_pattern, operator in pattern_parts:
            if operator == '+':
                minimal_case += base_pattern  # At least one
            elif operator == '?':
                # May or may not include (50% chance)
                if random.choice([True, False]):
                    minimal_case += base_pattern
            elif operator == '*':
                # May include 0-2 repetitions
                repeat_count = random.randint(0, 2)
                minimal_case += base_pattern * repeat_count
        cases.append(minimal_case)
        
        # 2. Typical valid case (moderate repetitions)
        typical_case = ""
        for base_pattern, operator in pattern_parts:
            if operator == '+':
                repeat_count = random.randint(1, 3)
                typical_case += base_pattern * repeat_count
            elif operator == '?':
                if random.choice([True, False]):
                    typical_case += base_pattern
            elif operator == '*':
                repeat_count = random.randint(1, 3)
                typical_case += base_pattern * repeat_count
        cases.append(typical_case)
        
        # 3. Maximum valid case (many repetitions)
        max_case = ""
        for base_pattern, operator in pattern_parts:
            if operator == '+':
                repeat_count = random.randint(2, 4)
                max_case += base_pattern * repeat_count
            elif operator == '?':
                max_case += base_pattern  # Always include
            elif operator == '*':
                repeat_count = random.randint(2, 4)
                max_case += base_pattern * repeat_count
        cases.append(max_case)
        
        # 4. Edge case: only first part matches
        if len(pattern_parts) > 1:
            first_base, first_op = pattern_parts[0]
            if first_op == '+':
                cases.append(first_base)
            elif first_op == '*':
                cases.append(first_base * 2)
        
        # 5. Edge case: only last part matches
        if len(pattern_parts) > 1:
            # Build case where only the last part has valid content
            last_valid_case = ""
            for i, (base_pattern, operator) in enumerate(pattern_parts):
                if i == len(pattern_parts) - 1:  # Last part
                    if operator == '+':
                        last_valid_case += base_pattern
                    elif operator == '*':
                        last_valid_case += base_pattern * 2
                    elif operator == '?':
                        last_valid_case += base_pattern
                else:  # Not last part
                    if operator == '?':
                        pass  # Don't include (0 occurrences)
                    elif operator == '*':
                        pass  # Don't include (0 occurrences)
                    elif operator == '+':
                        last_valid_case += base_pattern  # Must include at least one
            cases.append(last_valid_case)
        
        # ===== INVALID CASES =====
        
        # 6. Missing required parts (+ operator with 0 occurrences)
        for i, (base_pattern, operator) in enumerate(pattern_parts):
            if operator == '+':
                # Create case where this required part is missing
                invalid_case = ""
                for j, (bp, op) in enumerate(pattern_parts):
                    if i == j:
                        continue  # Skip the required part
                    if op == '+':
                        invalid_case += bp
                    elif op == '?':
                        invalid_case += bp
                    elif op == '*':
                        invalid_case += bp
                if invalid_case:  # Only add if non-empty
                    cases.append(invalid_case)
        
        # 7. Wrong order of parts
        if len(pattern_parts) >= 2:
            reversed_case = ""
            for base_pattern, operator in reversed(pattern_parts):
                if operator == '+':
                    reversed_case += base_pattern
                elif operator == '?':
                    reversed_case += base_pattern
                elif operator == '*':
                    reversed_case += base_pattern
            cases.append(reversed_case)
        
        # 8. Extra characters in between
        if len(pattern_parts) >= 2:
            interrupted_case = ""
            for i, (base_pattern, operator) in enumerate(pattern_parts):
                if operator == '+':
                    interrupted_case += base_pattern
                elif operator == '?':
                    interrupted_case += base_pattern
                elif operator == '*':
                    interrupted_case += base_pattern
                
                # Add interference except after last part
                if i < len(pattern_parts) - 1:
                    interference = random.choice(chars)
                    interrupted_case += interference
            cases.append(interrupted_case)
        
        # 9. Partial matches (incomplete patterns)
        for base_pattern, operator in pattern_parts:
            if len(base_pattern) > 1:
                partial = base_pattern[:-1]  # Remove last character
                padding = ''.join(random.choice(chars) for _ in range(3))
                cases.append(padding + partial + padding)
        
        # 10. Completely unrelated string
        unrelated_length = random.randint(5, 8)
        unrelated = ''.join(random.choice(chars) for _ in range(unrelated_length))
        cases.append(unrelated)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a regex pattern problem."""
        if params is None:
            # Generate random parameters using config
            concat_count = GeneratorConfig.REGEX_CONCATENATION_COUNT
            # Support both single value and range for REGEX_MAX_PATTERN_LENGTH
            if isinstance(GeneratorConfig.REGEX_MAX_PATTERN_LENGTH, (list, tuple)):
                min_pattern_len, max_pattern_len = GeneratorConfig.REGEX_MAX_PATTERN_LENGTH
            else:
                min_pattern_len, max_pattern_len = 1, GeneratorConfig.REGEX_MAX_PATTERN_LENGTH
            
            pattern_parts = []
            for _ in range(concat_count):
                base_pattern = self._generate_base_pattern(chars, max_pattern_len)
                operator = random.choice(GeneratorConfig.REGEX_OPERATORS)
                pattern_parts.append((base_pattern, operator))
        else:
            pattern_parts = params['pattern_parts']
        
        # Build the regex pattern
        regex_pattern = self._build_regex_pattern(pattern_parts)
        
        def pattern_func(s: str) -> bool:
            # Check if string matches the full regex pattern
            try:
                return bool(re.fullmatch(regex_pattern, s))
            except re.error:
                return False
        
        pattern_info = {"pattern_parts": pattern_parts}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        
        # Create human-readable description
        pattern_desc = []
        for base_pattern, operator in pattern_parts:
            if operator == '+':
                pattern_desc.append(f"({base_pattern})+")
            elif operator == '?':
                pattern_desc.append(f"({base_pattern})?")
            elif operator == '*':
                pattern_desc.append(f"({base_pattern})*")
        
        readable_pattern = ''.join(pattern_desc)
        name = f"Regex Pattern: {readable_pattern}"
        criteria = f"Accept if the tape matches the regex pattern '{readable_pattern}' exactly; reject otherwise. " \
                  f"'+' means one or more repetitions, '?' means zero or one occurrence, '*' means zero or more repetitions."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 