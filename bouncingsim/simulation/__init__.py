"""
Polygon Dynamics 2D Simulation Module

This module contains the core simulation functionality for 2D polygon dynamics,
including physics simulation and pygame-based visualization.
"""

from .core.physics_world import PhysicsWorld
from .core.ball import Ball
from .core.box import Box

__version__ = "1.0.0"
__all__ = [
    'PhysicsWorld',
    'Ball',
    'Box'
]
