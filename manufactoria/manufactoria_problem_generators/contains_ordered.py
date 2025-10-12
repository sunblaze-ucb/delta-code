"""
Contains Ordered Pattern Generator

Generates problems that require strings to contain a specific ordered subsequence.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class ContainsOrderedGenerator(BaseGenerator):
    """Generator for 'contains ordered subsequence' patterns."""
    
    def get_pattern_type(self) -> str:
        return "contains_ordered"
    
    def get_difficulty(self) -> str:
        return "medium"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for contains_ordered patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        min_len, max_len = GeneratorConfig.get_sequence_length_range('contains_ordered')
        return sum(len(chars) ** length for length in range(min_len, max_len + 1))
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for contains_ordered patterns."""
        min_len, max_len = GeneratorConfig.get_sequence_length_range('contains_ordered')
        sequences = self.get_sequence_space(chars, min_len, max_len)
        max_capacity = len(sequences)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} contains_ordered problems, but can only generate {max_capacity} unique sequences.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        params_list = []
        
        for i in range(count):
            sequence = sequences[i]  # No more cycling
            params_list.append({'sequence': sequence, 'ordered': True})
        
        return params_list
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for contains_ordered subsequence."""
        cases = []
        sequence = pattern_info.get("sequence", "")
        
        if not sequence:
            return cases
            
        # Get other characters not in sequence for building invalid cases
        other_chars = [c for c in chars if c not in sequence]
        
        # ===== VALID CASES =====
        
        # 1. Exact sequence (consecutive)
        cases.append(sequence)
        
        # 2. Sequence with single character insertions between each pair
        if len(sequence) > 1:
            for i in range(len(sequence) - 1):
                insert_char = random.choice(chars)
                case = sequence[:i+1] + insert_char + sequence[i+1:]
                cases.append(case)
        
        # 3. Sequence with characters interspersed (non-consecutive subsequence)
        if len(sequence) >= 2:
            interspersed = ""
            for i, char in enumerate(sequence):
                interspersed += char
                if i < len(sequence) - 1:  # Don't add after last character
                    interspersed += random.choice(chars)
            cases.append(interspersed)
        
        # 4. Sequence at beginning with suffix
        short_len, _ = GeneratorConfig.get_random_padding_length('short')
        suffix = ''.join(random.choice(chars) for _ in range(short_len))
        cases.append(sequence + suffix)
        
        # 5. Sequence at end with prefix
        medium_len, _ = GeneratorConfig.get_random_padding_length('medium')
        prefix = ''.join(random.choice(chars) for _ in range(medium_len))
        cases.append(prefix + sequence)
        
        # 6. Sequence in middle of long string (non-consecutive)
        if len(sequence) >= 2:
            long_case = ""
            # Add random prefix
            prefix_len = random.randint(2, 4)
            long_case += ''.join(random.choice(chars) for _ in range(prefix_len))
            
            # Add sequence characters with random gaps
            for i, char in enumerate(sequence):
                long_case += char
                if i < len(sequence) - 1:
                    gap_len = random.randint(1, 3)
                    long_case += ''.join(random.choice(chars) for _ in range(gap_len))
            
            # Add random suffix
            suffix_len = random.randint(2, 4)
            long_case += ''.join(random.choice(chars) for _ in range(suffix_len))
            cases.append(long_case)
        
        # 7. Multiple occurrences of sequence
        if len(sequence) >= 2:
            first_occurrence = sequence[:len(sequence)//2] + random.choice(chars) + sequence[len(sequence)//2:]
            middle_gap = ''.join(random.choice(chars) for _ in range(2))
            second_occurrence = sequence
            cases.append(first_occurrence + middle_gap + second_occurrence)
        
        # ===== INVALID CASES =====
        
        # 8. Missing first character
        if len(sequence) > 1:
            cases.append(sequence[1:])
        
        # 9. Missing last character
        if len(sequence) > 1:
            cases.append(sequence[:-1])
        
        # 10. Missing middle character
        if len(sequence) > 2:
            mid_idx = len(sequence) // 2
            cases.append(sequence[:mid_idx] + sequence[mid_idx+1:])
        
        # 11. Sequence in reverse order
        if len(sequence) >= 2:
            reversed_seq = sequence[::-1]
            padding_len = random.randint(2, 5)
            padding = ''.join(random.choice(chars) for _ in range(padding_len))
            cases.append(padding + reversed_seq + padding)
        
        # 12. Characters present but wrong order (complex case)
        if len(sequence) >= 3:
            # Take last character and put it first, keeping others in order
            wrong_order = sequence[-1] + sequence[:-1]
            gap_chars = ''.join(random.choice(chars) for _ in range(3))
            cases.append(gap_chars + wrong_order + gap_chars)
        
        # 13. Partial sequence with extra characters (missing key character)
        if len(sequence) > 2:
            partial = sequence[:-2] + sequence[-1]  # Skip second-to-last char
            long_len, _ = GeneratorConfig.get_random_padding_length('long')
            padding = ''.join(random.choice(chars) for _ in range(long_len))
            cases.append(padding + partial + padding)
        
        # 14. No sequence characters at all
        if other_chars:
            complex_len = GeneratorConfig.get_complex_case_length('medium_complex')
            no_seq_chars = ''.join(random.choice(other_chars) for _ in range(complex_len))
            cases.append(no_seq_chars)
        
        # 15. Almost complete sequence in long string (missing one crucial char)
        if len(sequence) >= 3:
            # Remove a middle character from sequence
            incomplete_seq = sequence[:len(sequence)//2] + sequence[len(sequence)//2 + 1:]
            long_len = GeneratorConfig.get_complex_case_length('large_complex')
            random_chars = ''.join(random.choice(chars) for _ in range(long_len))
            cases.append(random_chars + incomplete_seq + random_chars)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a contains_ordered problem."""
        if params is None:
            # Generate random parameters
            min_len, max_len = GeneratorConfig.get_sequence_length_range('contains_ordered')
            seq_length = random.randint(min_len, max_len)
            sequence = ''.join(random.choice(chars) for _ in range(seq_length))
        else:
            sequence = params['sequence']
        
        def pattern_func(s: str) -> bool:
            # Check if sequence appears as a subsequence (ordered but not necessarily consecutive)
            if not sequence:
                return True
            
            seq_idx = 0
            for char in s:
                if seq_idx < len(sequence) and char == sequence[seq_idx]:
                    seq_idx += 1
                    if seq_idx == len(sequence):
                        return True
            return seq_idx == len(sequence)
        
        pattern_info = {"sequence": sequence}
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        name = f"Contains {sequence}"
        criteria = f"Accept if the tape contains the subsequence {sequence} (does not have to be consecutive); reject otherwise."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        ) 