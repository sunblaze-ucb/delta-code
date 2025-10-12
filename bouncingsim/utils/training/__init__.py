"""
Training utilities for Polygon Dynamics 2D

This module contains training-related utilities and functions.
"""

from .polygon_dynamics_2d_utils import (
    check_distance,
    decode_tests,
    get_successful_tests_fast,
    get_successful_tests_stdio,
    run_tests_against_program_helper_2,
    run_individual_test_helper,
    should_execute,
    reliability_guard,
    partial_undo_reliability_guard
)

__version__ = "1.0.0"
__all__ = [
    'check_distance',
    'decode_tests',
    'get_successful_tests_fast',
    'get_successful_tests_stdio',
    'run_tests_against_program_helper_2',
    'run_individual_test_helper',
    'should_execute',
    'reliability_guard',
    'partial_undo_reliability_guard'
] 