"""
Utilities for Polygon Dynamics 2D

This module contains shared utilities, configuration, and helper functions.
"""

from .scene_loader import SceneLoader
from .training import check_distance
from .scene_to_prompt import generate_scene_messages
from .predict import predict_ball_positions
from .config import *

__version__ = "1.0.0"
__all__ = [
    'SceneLoader',
    'check_distance',
    'generate_scene_messages',
    'predict_ball_positions'
]
