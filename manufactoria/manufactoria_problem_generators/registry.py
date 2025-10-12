"""
Generator Registry System

Automatically discovers and manages all problem generators.
"""

import inspect
import pkgutil
import importlib
from typing import Dict, Type, List, Any
from .base import BaseGenerator


class GeneratorRegistry:
    """Registry for automatically discovering and managing generator classes."""
    
    _instance = None
    _generators: Dict[str, Type[BaseGenerator]] = {}
    _instantiated_generators: Dict[str, BaseGenerator] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, generator_class: Type[BaseGenerator]) -> Type[BaseGenerator]:
        """Register a generator class."""
        if not issubclass(generator_class, BaseGenerator):
            raise ValueError(f"Generator {generator_class.__name__} must inherit from BaseGenerator")
        
        # Get pattern type from an instance
        instance = generator_class()
        pattern_type = instance.get_pattern_type()
        
        cls._generators[pattern_type] = generator_class
        cls._instantiated_generators[pattern_type] = instance
        
        return generator_class
    
    @classmethod
    def get_generator(cls, pattern_type: str) -> BaseGenerator:
        """Get an instantiated generator by pattern type."""
        if pattern_type not in cls._instantiated_generators:
            raise ValueError(f"Unknown generator type: {pattern_type}. Available types: {list(cls._generators.keys())}")
        return cls._instantiated_generators[pattern_type]
    
    @classmethod
    def get_generator_class(cls, pattern_type: str) -> Type[BaseGenerator]:
        """Get a generator class by pattern type."""
        if pattern_type not in cls._generators:
            raise ValueError(f"Unknown generator type: {pattern_type}. Available types: {list(cls._generators.keys())}")
        return cls._generators[pattern_type]
    
    @classmethod
    def get_all_generators(cls) -> Dict[str, BaseGenerator]:
        """Get all instantiated generators."""
        return cls._instantiated_generators.copy()
    
    @classmethod
    def get_all_generator_classes(cls) -> Dict[str, Type[BaseGenerator]]:
        """Get all generator classes."""
        return cls._generators.copy()
    
    @classmethod
    def list_pattern_types(cls) -> List[str]:
        """List all available pattern types."""
        return list(cls._generators.keys())
    
    @classmethod
    def get_generator_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all generators."""
        info = {}
        for pattern_type, generator in cls._instantiated_generators.items():
            info[pattern_type] = {
                'class_name': cls._generators[pattern_type].__name__,
                'difficulty': generator.get_difficulty(),
                'two_color_capacity': generator.get_pattern_capacity('two_color'),
                'four_color_capacity': generator.get_pattern_capacity('four_color')
            }
        return info
    
    @classmethod
    def auto_discover(cls, verbose: bool = False):
        """Automatically discover and register all generators in the package.
        
        Args:
            verbose: If True, print discovery messages. Default is False.
        """
        import manufactoria_problem_generators
        
        # Get the package path
        package_path = manufactoria_problem_generators.__path__
        package_name = manufactoria_problem_generators.__name__
        
        # Modules to exclude from auto-discovery
        exclude_modules = {'base', 'registry', 'config', '__init__'}
        
        # Discover and import all modules in the package
        discovered_count = 0
        for importer, module_name, ispkg in pkgutil.iter_modules(package_path, f"{package_name}."):
            # Extract just the module name (without package prefix)
            short_name = module_name.split('.')[-1]
            
            # Skip excluded modules
            if short_name in exclude_modules:
                continue
                
            try:
                # Import the module to trigger registration
                importlib.import_module(module_name)
                discovered_count += 1
                if verbose:
                    print(f"Auto-discovered and imported: {module_name}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to import {module_name}: {e}")
        
        if verbose:
            print(f"Auto-discovery complete: {discovered_count} generator modules imported")
    
    @classmethod
    def clear_registry(cls):
        """Clear the registry (mainly for testing purposes)."""
        cls._generators.clear()
        cls._instantiated_generators.clear()


def register_generator(generator_class: Type[BaseGenerator]) -> Type[BaseGenerator]:
    """Decorator to register a generator class."""
    return GeneratorRegistry.register(generator_class)


# Initialize the registry as a singleton
registry = GeneratorRegistry() 