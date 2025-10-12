"""
Regex Same Number Generator

Generates problems that require strings to match patterns with specific numeric 
relationships between R and B characters, such as:
- R{n}B{n} (equal counts in sequence)
- B{2n}R{n} (2:1 ratio patterns)
- R{n}B{n}R{n} (symmetric patterns)
"""

import random
import re
from typing import List, Dict, Any, Tuple
from .base import BaseGenerator
from .config import GeneratorConfig
from .registry import register_generator


@register_generator
class RegexSameNumGenerator(BaseGenerator):
    """Generator for patterns with specific numeric relationships between R and B."""
    
    def __init__(self):
        super().__init__()
        # Possible values for x and y in terms of n: {n, n+1, n+2, n+3, 2n}
        self.variable_values = ['n', 'n+1', 'n+2', 'n+3', '2n']
        
        # Generate all combinations of pattern templates
        self.pattern_templates = []
        
        # R{x}B{y} patterns
        for x_val in self.variable_values:
            for y_val in self.variable_values:
                template = f"R{{{x_val}}}B{{{y_val}}}"
                self.pattern_templates.append((template, x_val, y_val, "sequence"))
        
        # B{x}R{y} patterns  
        for x_val in self.variable_values:
            for y_val in self.variable_values:
                template = f"B{{{x_val}}}R{{{y_val}}}"
                self.pattern_templates.append((template, x_val, y_val, "sequence"))
        
        # R{y}B{x}R{y} patterns
        for x_val in self.variable_values:
            for y_val in self.variable_values:
                template = f"R{{{y_val}}}B{{{x_val}}}R{{{y_val}}}"
                self.pattern_templates.append((template, x_val, y_val, "symmetric_r"))
        
        # B{x}R{y}B{x} patterns
        for x_val in self.variable_values:
            for y_val in self.variable_values:
                template = f"B{{{x_val}}}R{{{y_val}}}B{{{x_val}}}"
                self.pattern_templates.append((template, x_val, y_val, "symmetric_b"))
    
    def get_pattern_type(self) -> str:
        return "regex_same_num"
    
    def get_difficulty(self) -> str:
        return "hard"
    
    def get_pattern_capacity(self, color_mode: str) -> int:
        """Calculate capacity for same number patterns."""
        # Number of different pattern templates (all combinations)
        return len(self.pattern_templates)
    
    def generate_parameters(self, chars: List[str], count: int) -> List[Dict[str, Any]]:
        """Generate parameters for same number patterns based on generation mode."""
        # Use base class method to select templates based on generation mode
        selected_templates = self.select_parameters_by_mode(self.pattern_templates, count)
        
        params_list = []
        
        for pattern_template, x_val, y_val, pattern_type in selected_templates:
            params_list.append({
                'pattern_template': pattern_template,
                'x_val': x_val,
                'y_val': y_val,
                'pattern_type': pattern_type
            })
        
        return params_list
    
    def _evaluate_expression(self, expr: str, n: int) -> int:
        """Evaluate expressions like 'n', 'n+1', '2n' for given n."""
        if expr == 'n':
            return n
        elif expr == 'n+1':
            return n + 1
        elif expr == 'n+2':
            return n + 2
        elif expr == 'n+3':
            return n + 3
        elif expr == '2n':
            return 2 * n
        else:
            raise ValueError(f"Unknown expression: {expr}")
    
    def _parse_pattern_segments(self, template: str) -> List[Tuple[str, str]]:
        """Parse template like 'R{n}B{2n}' into [('R', 'n'), ('B', '2n')]."""
        import re
        # Find all segments like R{expr} or B{expr}
        segments = re.findall(r'([RB])\{([^}]+)\}', template)
        return segments
    
    def _check_pattern_match(self, input_str: str, template: str) -> bool:
        """Check if input string matches the template for any n >= 1."""
        if not input_str:
            return False
            
        segments = self._parse_pattern_segments(template)
        if not segments:
            return False
        
        # Try different values of n starting from 1
        max_n_to_try = min(50, len(input_str) + 1)  # Reasonable upper bound
        
        for n in range(1, max_n_to_try + 1):
            try:
                # Build expected pattern for this n
                expected_segments = []
                for char, expr in segments:
                    count = self._evaluate_expression(expr, n)
                    if count <= 0:
                        continue  # Skip invalid counts
                    expected_segments.append((char, count))
                
                # Check if input matches this pattern
                if self._matches_segments(input_str, expected_segments):
                    return True
                    
            except (ValueError, OverflowError):
                continue  # Skip invalid n values
        
        return False
    
    def _matches_segments(self, input_str: str, expected_segments: List[Tuple[str, int]]) -> bool:
        """Check if input string exactly matches the expected segments."""
        expected_str = ""
        for char, count in expected_segments:
            expected_str += char * count
        return input_str == expected_str
    
    def _get_pattern_specific_cases(self, pattern_type: str, chars: List[str], pattern_info: Dict) -> List[str]:
        """Generate pattern-specific test cases for same number patterns."""
        cases = []
        pattern_template = pattern_info.get("pattern_template", "")
        
        # ===== VALID CASES =====
        
        # Generate valid cases for different values of n
        for n in range(1, 5):  # Try n = 1, 2, 3, 4
            try:
                segments = self._parse_pattern_segments(pattern_template)
                expected_segments = []
                
                for char, expr in segments:
                    count = self._evaluate_expression(expr, n)
                    if count > 0 and count <= 20:  # Reasonable limits
                        expected_segments.append((char, count))
                
                if expected_segments:
                    valid_case = ""
                    for char, count in expected_segments:
                        valid_case += char * count
                    if len(valid_case) <= 30:  # Keep reasonable length
                        cases.append(valid_case)
                        
            except (ValueError, OverflowError):
                continue
        
        # ===== INVALID CASES =====
        
        # Generate some invalid cases by modifying valid ones
        if cases:
            valid_example = cases[0] if cases else "RB"
            
            # 1. Extra character at end
            cases.append(valid_example + random.choice(chars))
            
            # 2. Missing character from end
            if len(valid_example) > 1:
                cases.append(valid_example[:-1])
            
            # 3. Wrong order (reverse if multiple segments)
            segments = self._parse_pattern_segments(pattern_template)
            if len(segments) > 1:
                reversed_case = ""
                try:
                    for char, expr in reversed(segments):
                        count = self._evaluate_expression(expr, 2)  # Use n=2
                        if count > 0 and count <= 10:
                            reversed_case += char * count
                    if reversed_case and reversed_case != valid_example:
                        cases.append(reversed_case)
                except (ValueError, OverflowError):
                    pass
            
            # 4. Interleaved characters
            if len(valid_example) >= 4:
                interleaved = ""
                r_count = valid_example.count('R')
                b_count = valid_example.count('B')
                for i in range(max(r_count, b_count)):
                    if i < r_count:
                        interleaved += 'R'
                    if i < b_count:
                        interleaved += 'B'
                if interleaved != valid_example:
                    cases.append(interleaved)
        
        # 5. Wrong counts (off by one from a valid pattern)
        try:
            segments = self._parse_pattern_segments(pattern_template)
            if segments:
                wrong_count_case = ""
                for i, (char, expr) in enumerate(segments):
                    count = self._evaluate_expression(expr, 2)  # Use n=2
                    if i == 0:  # Modify first segment
                        count = max(1, count - 1)
                    if count > 0 and count <= 10:
                        wrong_count_case += char * count
                if wrong_count_case:
                    cases.append(wrong_count_case)
        except (ValueError, OverflowError):
            pass
        
        # 6. Empty string
        cases.append("")
        
        # 7. Single characters
        for char in chars:
            cases.append(char)
        
        # 8. Random strings
        for _ in range(3):
            random_length = random.randint(2, 8)
            random_string = ''.join(random.choice(chars) for _ in range(random_length))
            cases.append(random_string)
        
        return cases
    
    def generate_problem(self, chars: List[str], is_four_color: bool = False, 
                        params: Dict[str, Any] = None, index: int = 0) -> Dict[str, Any]:
        """Generate a same number pattern problem."""
        if params is None:
            # Generate random parameters
            template_idx = random.randint(0, len(self.pattern_templates) - 1)
            pattern_template, x_val, y_val, pattern_type = self.pattern_templates[template_idx]
        else:
            pattern_template = params['pattern_template']
            x_val = params['x_val']
            y_val = params['y_val']
            pattern_type = params['pattern_type']
        
        def pattern_func(s: str) -> bool:
            """Check if string matches the pattern for any valid n >= 1."""
            return self._check_pattern_match(s, pattern_template)
        
        pattern_info = {
            "pattern_template": pattern_template,
            "x_val": x_val,
            "y_val": y_val,
            "pattern_type": pattern_type
        }
        
        test_cases = self.create_test_cases(
            pattern_func, chars, False, self.get_pattern_type(), pattern_info
        )
        
        problem_id = str(__import__('uuid').uuid4())
        
        # Create human-readable description
        name = f"Variable Pattern: {pattern_template}"
        
        # Create description based on pattern structure
        description = self._create_pattern_description(pattern_template, x_val, y_val, pattern_type)
        criteria = f"Accept strings that match the pattern '{pattern_template}' for any n ≥ 1. {description}"
        
        return self.create_problem_dict(
            problem_id, name, criteria, test_cases, 
            self.get_difficulty(), self.get_pattern_type(), is_four_color
        )
    
    def _create_pattern_description(self, template: str, x_val: str, y_val: str, pattern_type: str) -> str:
        """Create a human-readable description of the pattern."""
        if pattern_type == "sequence":
            return f"A sequence where x={x_val} and y={y_val} for any positive integer n."
        elif pattern_type == "symmetric_r":
            return f"A symmetric pattern starting and ending with R, where x={x_val} and y={y_val} for any positive integer n."
        elif pattern_type == "symmetric_b":
            return f"A symmetric pattern starting and ending with B, where x={x_val} and y={y_val} for any positive integer n."
        else:
            return f"Pattern with x={x_val} and y={y_val} for any positive integer n." 