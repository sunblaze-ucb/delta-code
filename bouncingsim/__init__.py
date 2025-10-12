"""
Polygon Dynamics 2D - Physics Reasoning Dataset & LLM Testing Pipeline

A comprehensive physics simulation environment for testing Large Language Models (LLMs)
on 2D polygon dynamics reasoning tasks.

This package provides:
- Physics simulation with Box2D
- Scene generation and management
- Dataset generation for LLM testing
- LLM testing pipeline
- Interactive pygame visualization
"""

__version__ = "1.0.0"
__title__ = "Polygon Dynamics 2D"
__description__ = "Physics Reasoning Dataset & LLM Testing Pipeline"

# Import main modules for easy access
from . import simulation
from . import scene_generation
# Import main modules for easy access

__all__ = [
    'simulation',
    'scene_generation',
]
