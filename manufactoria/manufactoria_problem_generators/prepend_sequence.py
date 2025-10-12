"""
Prepend Sequence Pattern Generator with Mutations

Generates problems that require applying a mutation operation to the input tape,
then prepending a specific sequence to the beginning of the result.

Mutation operations:
1. keep color unchanged
2. change B to [pattern]
3. change R to [pattern]
4. remove B
5. remove R
6. swap R&B

The final operation is to prepend a sequence to the beginning of the mutated tape.
"""

import random
from typing import List, Dict, Any
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class PrependSequenceGenerator(BaseGenerator):
    """Generator for 'prepend sequence with mutations' patterns."""
    
    def get_pattern_type(self) -> str:
        return "prepend_sequence"
    
    def get_difficulty(self) -> str:
        return "basic"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for prepend_sequence patterns."""
        chars = self.two_color_chars if color_mode == 'two_color' else self.four_color_chars
        min_len, max_len = GeneratorConfig.get_sequence_length_range('prepend_sequence')
        
        # Number of possible sequences to prepend
        num_sequences = sum(len(chars) ** length for length in range(min_len, max_len + 1))
        
        # For mutations that use patterns (change B/R to pattern), calculate pattern space
        pattern_space = sum(len(chars) ** length for length in range(1, 3))  # Patterns of length 1-3
        
        # Total capacity = mutations without patterns + mutations with patterns * pattern space
        # Mutations without patterns: keep unchanged, remove B, remove R, swap R&B (4 operations)
        # Mutations with patterns: change B to pattern, change R to pattern (2 operations)
        total_capacity = (4 + 2 * pattern_space) * num_sequences
        
        return total_capacity
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate systematic parameters for prepend_sequence patterns."""
        min_len, max_len = GeneratorConfig.get_sequence_length_range('prepend_sequence')
        sequences = self.get_sequence_space(chars, min_len, max_len)
        
        # Check mutation complexity level
        enable_mutations = getattr(GeneratorConfig, 'PREPEND_ENABLE_MUTATIONS', True)
        pattern_mutations = getattr(GeneratorConfig, 'PREPEND_PATTERN_MUTATIONS', False)
        
        if not enable_mutations:
            # Only use keep_unchanged (no mutations)
            mutations = [
                {"type": "keep_unchanged", "pattern": None}
            ]
        elif pattern_mutations:
            # Hard tier: Pattern-to-pattern mutations
            mutations = []
            
            # Generate source patterns (2-character) and target patterns (1-2 character)
            source_patterns = []
            target_patterns = []
            
            # Source patterns: 2-character sequences
            for length in range(2, 3):  # 2-character patterns
                patterns = self.get_sequence_space(chars, length, length)
                source_patterns.extend(patterns)
            
            # Target patterns: empty string, 1-2 character sequences  
            target_patterns.append("")  # Add empty string (deletion)
            for length in range(1, 3):  # 1-2 character patterns
                patterns = self.get_sequence_space(chars, length, length)
                target_patterns.extend(patterns)
            
            # Add pattern-to-pattern mutations
            for source_pattern in source_patterns:
                for target_pattern in target_patterns:
                    if source_pattern != target_pattern:  # Don't replace with same pattern
                        mutations.append({
                            "type": "replace_pattern_to_pattern", 
                            "source_pattern": source_pattern,
                            "target_pattern": target_pattern
                        })
        else:
            # Medium tier: Single character mutations (full set)
            mutations = [
                {"type": "remove_B", "pattern": None},
                {"type": "remove_R", "pattern": None},
                {"type": "swap_RB", "pattern": None}
            ]
            
            # Add single character to pattern mutations
            pattern_lengths = [1, 2]  # Keep patterns short
            for pattern_len in pattern_lengths:
                for length in range(1, pattern_len + 1):
                    for pattern in self.get_sequence_space(chars, length, length):
                        mutations.append({"type": "change_B_to_pattern", "pattern": pattern})
                        mutations.append({"type": "change_R_to_pattern", "pattern": pattern})
        
        max_capacity = len(mutations) * len(sequences)
        
        # Warn if requested count exceeds capacity
        if count > max_capacity:
            print(f"Warning: Requested {count} prepend_sequence problems, but can only generate {max_capacity} unique combinations.")
            print(f"Generating {max_capacity} unique problems instead to avoid duplicates.")
            count = max_capacity
        
        params_list = []
        
        for i in range(count):
            mutation_idx = i % len(mutations)
            sequence_idx = (i // len(mutations)) % len(sequences)
            
            mutation = mutations[mutation_idx]
            sequence = sequences[sequence_idx]
            
            if mutation["type"] == "replace_pattern_to_pattern":
                params_list.append({
                    'prepend_sequence': sequence,
                    'mutation_type': mutation["type"],
                    'source_pattern': mutation["source_pattern"],
                    'target_pattern': mutation["target_pattern"]
                })
            else:
                params_list.append({
                    'prepend_sequence': sequence,
                    'mutation_type': mutation["type"],
                    'mutation_pattern': mutation.get("pattern")
                })
        
        return params_list
    
    def _apply_mutation(self, input_str: str, mutation_type: str, mutation_pattern: str = None, 
                       source_pattern: str = None, target_pattern: str = None) -> str:
        """Apply the specified mutation to the input string."""
        if mutation_type == "keep_unchanged":
            return input_str
        elif mutation_type == "change_B_to_pattern":
            return input_str.replace('B', mutation_pattern or '')
        elif mutation_type == "change_R_to_pattern":
            return input_str.replace('R', mutation_pattern or '')
        elif mutation_type == "remove_B":
            return input_str.replace('B', '')
        elif mutation_type == "remove_R":
            return input_str.replace('R', '')
        elif mutation_type == "swap_RB":
            result = ""
            for char in input_str:
                if char == 'R':
                    result += 'B'
                elif char == 'B':
                    result += 'R'
                else:
                    result += char
            return result
        elif mutation_type == "replace_pattern_to_pattern":
            return input_str.replace(source_pattern, target_pattern)
        else:
            return input_str
    
    def _get_mutation_description(self, mutation_type: str, mutation_pattern: str = None, 
                                 source_pattern: str = None, target_pattern: str = None) -> str:
        """Get human-readable description of the mutation."""
        if mutation_type == "keep_unchanged":
            return "Keep color unchanged"
        elif mutation_type == "change_B_to_pattern":
            return f"Change B to {mutation_pattern}"
        elif mutation_type == "change_R_to_pattern":
            return f"Change R to {mutation_pattern}"
        elif mutation_type == "remove_B":
            return "Remove B"
        elif mutation_type == "remove_R":
            return "Remove R"
        elif mutation_type == "swap_RB":
            return "Swap color R&B"
        elif mutation_type == "replace_pattern_to_pattern":
            if target_pattern == "":
                return f"Remove patterns {source_pattern}"
            else:
                return f"Replace patterns {source_pattern} with {target_pattern}"
        else:
            return "unknown mutation"
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for prepend_sequence."""
        cases = []
        
        # Basic cases - these should be accepted by the pattern function
        cases.append("")  # Empty input
        cases.append(chars[0])  # Single character input
        cases.append(chars[1] if len(chars) > 1 else chars[0])  # Different single character
        
        # Two character inputs
        if len(chars) >= 2:
            cases.extend([chars[0] + chars[1], chars[1] + chars[0]])
            cases.extend([chars[0] + chars[0], chars[1] + chars[1]])
        
        # Medium length inputs
        medium_input = ''.join(random.choice(chars) for _ in range(4))
        cases.append(medium_input)
        
        # Inputs with patterns that will be affected by mutations
        cases.append("BBB")  # Multiple Bs
        cases.append("RRR")  # Multiple Rs
        cases.append("BRBR")  # Alternating
        cases.append("RBRB")  # Alternating reversed
        
        # Longer inputs
        long_input = ''.join(random.choice(chars) for _ in range(8))
        cases.append(long_input)
        
        # Complex cases
        complex_len = GeneratorConfig.get_complex_case_length('medium_complex')
        complex_input = ''.join(random.choice(chars) for _ in range(complex_len))
        cases.append(complex_input)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a prepend_sequence problem with mutations."""
        if params is None:
            # Generate random parameters
            min_len, max_len = GeneratorConfig.get_sequence_length_range('prepend_sequence')
            seq_length = random.randint(min_len, max_len)
            prepend_sequence = ''.join(random.choice(chars) for _ in range(seq_length))
            
            # Random mutation
            mutations = ["keep_unchanged", "remove_B", "remove_R", "swap_RB"]
            pattern_mutations = ["change_B_to_pattern", "change_R_to_pattern"]
            
            if random.choice([True, False]):
                mutation_type = random.choice(mutations)
                mutation_pattern = None
            else:
                mutation_type = random.choice(pattern_mutations)
                pattern_len = random.randint(1, 2)
                mutation_pattern = ''.join(random.choice(chars) for _ in range(pattern_len))
        else:
            prepend_sequence = params['prepend_sequence']
            mutation_type = params['mutation_type']
            mutation_pattern = params.get('mutation_pattern')
            source_pattern = params.get('source_pattern')
            target_pattern = params.get('target_pattern')
        
        def pattern_func(s: str) -> bool:
            """For prepend_sequence, we accept ALL valid input strings."""
            return True
        
        def expected_output_func(s: str) -> str:
            """Return the expected output: prepended sequence + mutated input."""
            mutated = self._apply_mutation(s, mutation_type, mutation_pattern, source_pattern, target_pattern)
            return prepend_sequence + mutated
        
        pattern_info = {
            "prepend_sequence": prepend_sequence,
            "mutation_type": mutation_type,
            "mutation_pattern": mutation_pattern,
            "source_pattern": source_pattern,
            "target_pattern": target_pattern
        }
        
        # Create test cases with output checking enabled
        test_cases = []
        seen_inputs = set()
        
        # Generate deterministic test cases
        deterministic_cases = self._generate_deterministic_test_cases('prepend_sequence', chars, pattern_info)
        
        # Add deterministic cases with expected outputs
        for test_input in deterministic_cases:
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_output = expected_output_func(test_input)
                mutated_input = self._apply_mutation(test_input, mutation_type, mutation_pattern, source_pattern, target_pattern)
                test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                    "expected_accepted": True,  # All inputs should be accepted
                    "check_output": True,  # We check the output
                    "description": f"Input '{test_input}' → mutated: '{mutated_input}' → output: '{expected_output}'"
                })
        
        # Add more test cases to reach minimum requirements
        while len(test_cases) < GeneratorConfig.MIN_ACCEPTING_CASES + GeneratorConfig.MIN_REJECTING_CASES:
            test_input = self.generate_test_string(chars)
            if test_input not in seen_inputs:
                seen_inputs.add(test_input)
                expected_output = expected_output_func(test_input)
                mutated_input = self._apply_mutation(test_input, mutation_type, mutation_pattern, source_pattern, target_pattern)
                test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                    "expected_accepted": True,
                    "check_output": True,
                    "description": f"Input '{test_input}' → mutated: '{mutated_input}' → output: '{expected_output}'"
                })
        
        # Limit to max test cases
        test_cases = test_cases[:GeneratorConfig.MAX_TEST_CASES]
        
        problem_id = str(__import__('uuid').uuid4())
        
        # Create descriptive name and criteria
        # Check if this is effectively a no-op mutation
        is_no_op = (mutation_type == "keep_unchanged" or 
                   (mutation_type == "change_B_to_pattern" and mutation_pattern == "B") or
                   (mutation_type == "change_R_to_pattern" and mutation_pattern == "R"))
        
        if is_no_op:
            name = f"Prepend {prepend_sequence}"
            criteria = f"Put '{prepend_sequence}' at the beginning of the tape."
            description = f"The machine should read any input sequence and prepend '{prepend_sequence}' to the beginning."
        else:
            mutation_desc = self._get_mutation_description(mutation_type, mutation_pattern, source_pattern, target_pattern)
            name = f"Prepend {prepend_sequence} after {mutation_desc}"
            criteria = f"{mutation_desc}, then put '{prepend_sequence}' at the beginning of the tape."
            description = f"The machine should read any input sequence, apply the mutation ({mutation_desc}), then prepend '{prepend_sequence}' to the beginning of the result."
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color,
            description=description
        )