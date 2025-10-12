"""
Utility functions for parsing mathematical expressions in configuration files.
"""
import math
import re


def parse_numeric_value(value, default=0.0):
    """
    Parse a numeric value that can be either a number or a string expression.
    
    Supports mathematical expressions with:
    - Basic arithmetic operators: +, -, *, /, **
    - Constants: pi, e
    - Functions: sin, cos, tan, sqrt, etc.
    
    Args:
        value: Either a number (int/float) or a string expression (e.g., "2*pi/3")
        default: Default value to return if value is None
        
    Returns:
        float: The evaluated numeric value
        
    Examples:
        >>> parse_numeric_value(1.5)
        1.5
        >>> parse_numeric_value("2*pi/3")
        2.0943951023931953
        >>> parse_numeric_value("pi/2")
        1.5707963267948966
    """
    if value is None:
        return default
    
    # If it's already a number, return it
    if isinstance(value, (int, float)):
        return float(value)
    
    # If it's a string, try to evaluate it
    if isinstance(value, str):
        try:
            # Create a safe namespace with math functions and constants
            safe_namespace = {
                'pi': math.pi,
                'e': math.e,
                'tau': math.tau,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'sqrt': math.sqrt,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'abs': abs,
                'pow': pow,
                '__builtins__': {}  # Disable built-in functions for security
            }
            
            # Evaluate the expression safely
            result = eval(value, safe_namespace, {})
            return float(result)
        except (ValueError, SyntaxError, NameError, TypeError) as e:
            raise ValueError(f"Failed to parse numeric expression '{value}': {e}")
    
    raise TypeError(f"Value must be a number or string expression, got {type(value)}")


def parse_rotation_speed(config_dict, key="rotation_speed", default=0.0):
    """
    Helper function to parse rotation_speed from a configuration dictionary.
    
    Args:
        config_dict: Dictionary containing the configuration
        key: Key to look up in the dictionary (default: "rotation_speed")
        default: Default value if key is not present
        
    Returns:
        float: The parsed rotation speed value
    """
    value = config_dict.get(key, default)
    return parse_numeric_value(value, default)

