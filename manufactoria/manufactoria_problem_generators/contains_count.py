"""
Contains Count Pattern Generator

Generates problems that require strings to contain specific counts of multiple characters.
"""

import random
from typing import List, Dict, Any, Tuple
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class ContainsCountGenerator(BaseGenerator):
    """Generator for 'contains count' patterns with multiple character requirements."""
    
    def get_pattern_type(self) -> str:
        return "contains_count"
    
    def get_difficulty(self) -> str:
        return "medium"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for contains_count patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        
        # Support both single value and range for COUNT_THRESHOLDS
        if isinstance(GeneratorConfig.COUNT_THRESHOLDS, (list, tuple)):
            min_total_count, max_total_count = GeneratorConfig.COUNT_THRESHOLDS
        else:
            min_total_count, max_total_count = 1, GeneratorConfig.COUNT_THRESHOLDS
            
        num_chars = len(chars)
        max_chars_in_pattern = min(3, num_chars)  # Only consider up to 3 characters maximum
        
        total_combinations = 0
        
        # For each possible number of characters involved (1 to max 3)
        for num_involved_chars in range(1, max_chars_in_pattern + 1):
            # Choose which characters to involve
            from math import comb
            char_selection_ways = comb(num_chars, num_involved_chars)
            
            # Count ways to distribute counts among the selected characters
            # This is the number of positive integer solutions to:
            # x1 + x2 + ... + x_num_involved_chars = k, for k from max(num_involved_chars, min_total_count) to max_total_count
            # where each xi >= 1
            count_distribution_ways = 0
            for total_count in range(max(num_involved_chars, min_total_count), max_total_count + 1):
                # Stars and bars: distribute (total_count - num_involved_chars) among num_involved_chars positions
                # This gives us (total_count - 1) choose (num_involved_chars - 1)
                if total_count - 1 >= 0 and num_involved_chars - 1 >= 0:
                    count_distribution_ways += comb(total_count - 1, num_involved_chars - 1)
            
            total_combinations += char_selection_ways * count_distribution_ways
        
        return total_combinations
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for contains_count patterns."""
        # Support both single value and range for COUNT_THRESHOLDS
        if isinstance(GeneratorConfig.COUNT_THRESHOLDS, (list, tuple)):
            min_total_count, max_total_count = GeneratorConfig.COUNT_THRESHOLDS
        else:
            min_total_count, max_total_count = 1, GeneratorConfig.COUNT_THRESHOLDS
        params_list = []
        
        # Generate all valid combinations systematically
        all_combinations = []
        
        # Single character requirements
        for char in chars:
            for char_count in range(max(1, min_total_count), max_total_count + 1):
                all_combinations.append([(char, char_count)])
        
        # Two character requirements
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                for count1 in range(1, max_total_count):
                    for count2 in range(1, max_total_count - count1 + 1):
                        total_count = count1 + count2
                        if min_total_count <= total_count <= max_total_count:
                            all_combinations.append([
                                (chars[i], count1),
                                (chars[j], count2)
                            ])
        
        # Three character requirements (only if we have enough chars and budget)
        if len(chars) >= 3:
            for i in range(len(chars)):
                for j in range(i + 1, len(chars)):
                    for k in range(j + 1, len(chars)):
                        for count1 in range(1, max_total_count - 1):
                            for count2 in range(1, max_total_count - count1):
                                for count3 in range(1, max_total_count - count1 - count2 + 1):
                                    total_count = count1 + count2 + count3
                                    if min_total_count <= total_count <= max_total_count:
                                        all_combinations.append([
                                            (chars[i], count1),
                                            (chars[j], count2),
                                            (chars[k], count3)
                                        ])
        
        # Four character requirements (only for four-color mode with minimal counts)
        # if len(chars) >= 4 and max_total_count >= 4:
        #     for i in range(len(chars)):
        #         for j in range(i + 1, len(chars)):
        #             for k in range(j + 1, len(chars)):
        #                 for l in range(k + 1, len(chars)):
        #                     # Only allow minimal counts for 4 characters
        #                     if 4 <= max_total_count:
        #                         all_combinations.append([
        #                             (chars[i], 1),
        #                             (chars[j], 1),
        #                             (chars[k], 1),
        #                             (chars[l], 1)
        #                         ])
        
        # Use base class method to select combinations based on generation mode
        selected_combinations = self.select_parameters_by_mode(all_combinations, count)
        
        for requirements in selected_combinations:
            params_list.append({
                'requirements': requirements,  # List of (char, min_count) tuples
                'ordered': False
            })
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for contains_count."""
        cases = []
        requirements = pattern_info.get("requirements", [])
        
        if not requirements:
            return cases
        
        # Get all required characters and their counts
        required_chars = {char: count for char, count in requirements}
        other_chars = [c for c in chars if c not in required_chars]
        
        # ===== VALID CASES =====
        
        # 1. Exact requirements (minimal case)
        exact_case = ""
        for char, count in requirements:
            exact_case += char * count
        # Shuffle to make it unordered
        exact_chars = list(exact_case)
        random.shuffle(exact_chars)
        cases.append(''.join(exact_chars))
        
        # 2. Exact requirements + extra occurrences of required chars
        extra_case = exact_case
        for char, count in requirements:
            extra_case += char * random.randint(1, 2)
        extra_chars = list(extra_case)
        random.shuffle(extra_chars)
        cases.append(''.join(extra_chars))
        
        # 3. Requirements + other characters mixed in
        if other_chars:
            mixed_case = exact_case
            mixed_case += ''.join(random.choice(other_chars) for _ in range(3))
            mixed_chars = list(mixed_case)
            random.shuffle(mixed_chars)
            cases.append(''.join(mixed_chars))
        
        # 4. Requirements in a longer string
        long_case = ""
        # Add required chars with sufficient counts
        for char, count in requirements:
            long_case += char * (count + random.randint(0, 2))
        # Add padding
        padding_len = GeneratorConfig.get_complex_case_length('medium_complex') - len(long_case)
        if padding_len > 0 and other_chars:
            padding = ''.join(random.choice(chars) for _ in range(padding_len))
            long_case += padding
        long_chars = list(long_case)
        random.shuffle(long_chars)
        cases.append(''.join(long_chars))
        
        # 5. Multiple occurrences spread throughout
        very_long_case = ""
        # First, ensure minimum requirements
        for char, count in requirements:
            very_long_case += char * count
        # Then add more characters including required ones
        extra_len = GeneratorConfig.get_complex_case_length('large_complex') - len(very_long_case)
        if extra_len > 0:
            extra_chars_list = []
            for _ in range(extra_len):
                # Bias towards adding more required chars and some others
                if random.random() < 0.6 and requirements:
                    char, _ = random.choice(requirements)
                    extra_chars_list.append(char)
                else:
                    extra_chars_list.append(random.choice(chars))
            very_long_case += ''.join(extra_chars_list)
        very_long_chars = list(very_long_case)
        random.shuffle(very_long_chars)
        cases.append(''.join(very_long_chars))
        
        # ===== INVALID CASES =====
        
        # 6. Missing one required character entirely
        if len(requirements) > 1:
            missing_char_case = ""
            for i, (char, count) in enumerate(requirements[:-1]):  # Skip last requirement
                missing_char_case += char * count
            if other_chars:
                missing_char_case += ''.join(random.choice(other_chars) for _ in range(5))
            missing_chars = list(missing_char_case)
            random.shuffle(missing_chars)
            cases.append(''.join(missing_chars))
        
        # 7. One character count insufficient
        insufficient_case = ""
        for i, (char, count) in enumerate(requirements):
            if i == 0 and count > 1:
                # Make first requirement insufficient
                insufficient_case += char * (count - 1)
            else:
                insufficient_case += char * count
        if other_chars:
            insufficient_case += ''.join(random.choice(other_chars) for _ in range(3))
        insufficient_chars = list(insufficient_case)
        random.shuffle(insufficient_chars)
        cases.append(''.join(insufficient_chars))
        
        # 8. No required characters at all
        if other_chars:
            no_req_len = GeneratorConfig.get_complex_case_length('medium_complex')
            no_req_case = ''.join(random.choice(other_chars) for _ in range(no_req_len))
            cases.append(no_req_case)
        
        # 9. All required chars present but all insufficient counts
        all_insufficient_case = ""
        for char, count in requirements:
            if count > 1:
                all_insufficient_case += char * (count - 1)
            # If count is 1, we skip adding this char (making it 0)
        if other_chars:
            padding_len = max(5, GeneratorConfig.get_complex_case_length('medium_complex') - len(all_insufficient_case))
            all_insufficient_case += ''.join(random.choice(other_chars) for _ in range(padding_len))
        all_insufficient_chars = list(all_insufficient_case)
        random.shuffle(all_insufficient_chars)
        cases.append(''.join(all_insufficient_chars))
        
        # 10. Long string with almost sufficient counts
        almost_case = ""
        for char, count in requirements:
            # One requirement will be just short
            if char == requirements[0][0] and count > 1:
                almost_case += char * (count - 1)
            else:
                almost_case += char * count
        long_padding_len = GeneratorConfig.get_complex_case_length('large_complex') - len(almost_case)
        if long_padding_len > 0:
            almost_case += ''.join(random.choice(chars) for _ in range(long_padding_len))
        almost_chars = list(almost_case)
        random.shuffle(almost_chars)
        cases.append(''.join(almost_chars))
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a contains_count problem with multiple character requirements."""
        # Support both single value and range for COUNT_THRESHOLDS
        if isinstance(GeneratorConfig.COUNT_THRESHOLDS, (list, tuple)):
            min_total_count, max_total_count = GeneratorConfig.COUNT_THRESHOLDS
        else:
            min_total_count, max_total_count = 1, GeneratorConfig.COUNT_THRESHOLDS
        
        requirements = params['requirements']
        # Validate that total count is within the valid range
        total_count = sum(count for _, count in requirements)
        if not (min_total_count <= total_count <= max_total_count):
            raise ValueError(f"Total count {total_count} is outside the valid range [{min_total_count}, {max_total_count}]")
        
        def pattern_func(s: str) -> bool:
            """Check if string satisfies all character count requirements."""
            for char, min_count in requirements:
                if s.count(char) < min_count:
                    return False
            return True
        
        pattern_info = {"requirements": requirements}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        
        # Generate readable name and criteria
        if len(requirements) == 1:
            char, count = requirements[0]
            name = f"At least {count} {char}{'s' if count > 1 else ''}"
            criteria = f"Accept if the tape has at least {count} {char.lower()}{'s' if count > 1 else ''}; reject otherwise."
        else:
            # Multiple requirements
            requirement_strs = []
            criteria_parts = []
            for char, count in requirements:
                requirement_strs.append(f"{count} {char}{'s' if count > 1 else ''}")
                criteria_parts.append(f"at least {count} {char.lower()}{'s' if count > 1 else ''}")
            
            name = f"Need {' and '.join(requirement_strs)}"
            criteria = f"Accept if the tape has {' and '.join(criteria_parts)}; reject otherwise."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 