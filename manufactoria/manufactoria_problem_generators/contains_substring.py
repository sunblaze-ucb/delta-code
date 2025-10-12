"""
Contains Substring Pattern Generator

Generates problems that require strings to contain a specific consecutive substring.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class ContainsSubstringGenerator(BaseGenerator):
    """Generator for 'contains substring' patterns (consecutive characters)."""
    
    def get_pattern_type(self) -> str:
        return "contains_substring"
    
    def get_difficulty(self) -> str:
        return "easy"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for contains_substring patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        min_len, max_len = GeneratorConfig.get_sequence_length_range('contains_substring')
        return sum(len(chars) ** length for length in range(min_len, max_len + 1))
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for contains_substring patterns."""
        min_len, max_len = GeneratorConfig.get_sequence_length_range('contains_substring')
        sequences = self.get_sequence_space(chars, min_len, max_len)
        max_capacity = len(sequences)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} contains_substring problems, but can only generate {max_capacity} unique sequences.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        params_list = []
        
        for i in range(count):
            sequence = sequences[i]  # No more cycling
            params_list.append({'sequence': sequence, 'consecutive': True})
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for contains_substring (consecutive)."""
        cases = []
        sequence = pattern_info.get("sequence", "")
        
        if not sequence:
            return cases
            
        # Get other characters not in sequence for building invalid cases
        other_chars = [c for c in chars if c not in sequence]
        
        # ===== VALID CASES =====
        
        # 1. Exact substring (the sequence itself)
        cases.append(sequence)
        
        # 2. Substring at beginning with suffix
        short_len, _ = GeneratorConfig.get_random_padding_length('short')
        suffix = ''.join(random.choice(chars) for _ in range(short_len))
        cases.append(sequence + suffix)
        
        # 3. Substring at end with prefix
        medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
        prefix = ''.join(random.choice(chars) for _ in range(medium_len))
        cases.append(prefix + sequence)
        
        # 4. Substring in middle of string
        short_prefix = ''.join(random.choice(chars) for _ in range(random.randint(1, 3)))
        short_suffix = ''.join(random.choice(chars) for _ in range(random.randint(1, 3)))
        cases.append(short_prefix + sequence + short_suffix)
        
        # 5. Multiple occurrences of substring
        gap = ''.join(random.choice(chars) for _ in range(random.randint(1, 3)))
        cases.append(sequence + gap + sequence)
        
        # 6. Substring in long string
        # Use shorter lengths for better performance
        prefix_len = random.randint(3, 5)
        suffix_len = random.randint(3, 5)
        long_prefix = ''.join(random.choice(chars) for _ in range(prefix_len))
        long_suffix = ''.join(random.choice(chars) for _ in range(suffix_len))
        cases.append(long_prefix + sequence + long_suffix)
        
        # 7. Overlapping occurrences (if possible)
        if len(sequence) >= 2 and len(set(sequence)) < len(sequence):
            # For sequences with repeated characters, create overlapping pattern
            overlap_case = sequence + sequence[1:]
            cases.append(overlap_case)
        
        # ===== INVALID CASES =====
        
        # 8. Characters of sequence separated by other characters
        if len(sequence) >= 2:
            separated = ""
            for i, char in enumerate(sequence):
                separated += char
                if i < len(sequence) - 1:
                    separated += random.choice(chars)
            cases.append(separated)
        
        # 9. Partial substring (missing characters from end)
        if len(sequence) > 1:
            for i in range(1, len(sequence)):
                partial = sequence[:i]
                padding_len = random.randint(3, 6)
                padding = ''.join(random.choice(chars) for _ in range(padding_len))
                cases.append(padding + partial + padding)
        
        # 10. Reversed substring
        if len(sequence) >= 2:
            reversed_seq = sequence[::-1]
            padding_len = random.randint(2, 4)
            padding = ''.join(random.choice(chars) for _ in range(padding_len))
            cases.append(padding + reversed_seq + padding)
        
        # 11. Characters present but in different order
        if len(sequence) >= 3:
            # Simple reordering to avoid infinite loops
            shuffled_chars = list(sequence)
            if len(set(sequence)) > 1:  # Only shuffle if there are different characters
                # Safe reordering: just reverse the string
                shuffled_chars.reverse()
                # If reverse gives same result, swap first two chars
                if ''.join(shuffled_chars) == sequence and len(shuffled_chars) >= 2:
                    shuffled_chars[0], shuffled_chars[1] = shuffled_chars[1], shuffled_chars[0]
            padding = ''.join(random.choice(chars) for _ in range(3))
            cases.append(padding + ''.join(shuffled_chars) + padding)
        
        # 12. Almost complete substring with interruption
        if len(sequence) >= 3:
            mid_idx = len(sequence) // 2
            interrupted = sequence[:mid_idx] + random.choice(chars) + sequence[mid_idx:]
            cases.append(interrupted)
        
        # 13. No characters from sequence at all
        if other_chars:
            no_seq_len = GeneratorConfig.get_complex_case_length('medium_complex')
            no_seq_chars = ''.join(random.choice(other_chars) for _ in range(no_seq_len))
            cases.append(no_seq_chars)
        
        # 14. Sequence with extra character inserted in middle
        if len(sequence) >= 2:
            insert_pos = random.randint(1, len(sequence) - 1)
            extra_char = random.choice(chars)
            broken_seq = sequence[:insert_pos] + extra_char + sequence[insert_pos:]
            padding = ''.join(random.choice(chars) for _ in range(3))  # Reduced padding
            cases.append(padding + broken_seq + padding)
        
        # 15. Long string with all characters present but never consecutive
        if len(sequence) >= 2:
            # Simplified version to avoid performance issues
            spread_out = ""
            remaining_chars = list(sequence)
            
            # Distribute sequence characters with single separators (much more efficient)
            for i, char in enumerate(remaining_chars):
                spread_out += char
                if i < len(remaining_chars) - 1:  # Add separator except after last char
                    spread_out += random.choice(chars)
            
            # Add some padding
            prefix = ''.join(random.choice(chars) for _ in range(2))
            suffix = ''.join(random.choice(chars) for _ in range(2))
            cases.append(prefix + spread_out + suffix)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a contains_substring problem."""
        if params is None:
            # Generate random parameters
            min_len, max_len = GeneratorConfig.get_sequence_length_range('contains_substring')
            seq_length = random.randint(min_len, max_len)
            sequence = ''.join(random.choice(chars) for _ in range(seq_length))
        else:
            sequence = params['sequence']
        
        def pattern_func(s: str) -> bool:
            # Check if sequence appears as a substring (consecutive characters)
            return sequence in s
        
        pattern_info = {"sequence": sequence}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        name = f"Contains '{sequence}'"
        criteria = f"Accept if the tape contains the substring '{sequence}' (must be consecutive); reject otherwise."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 