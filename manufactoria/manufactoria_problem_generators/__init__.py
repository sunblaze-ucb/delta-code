"""
Modular Problem Generators for Manufactoria

This package contains individual generators for different problem types.
Uses a registry system for automatic discovery and management of generators.
"""

from .config import GeneratorConfig
from .base import BaseGenerator
from .registry import GeneratorRegistry, register_generator

# Auto-discover and register all generators
GeneratorRegistry.auto_discover()

# Export the registry and key classes
__all__ = [
    'GeneratorConfig',
    'BaseGenerator', 
    'GeneratorRegistry',
    'register_generator'
] 