"""
Main Source Package Module

This is the main source code package for SPINOTOR — BLDC motor control simulator.

Includes:
- core: Motor model, simulation engine, load profiles
- control: Control blocks (SVM, V/f, FOC ready)
- ui: GUI application with accessibility support
- utils: Configuration and utilities
- visualization: Plotting and data export

:author: Amine Khettat — Blind Systems
:version: 0.12.0
"""

from importlib import import_module

__all__ = [
    "control",
    "core",
    "hardware",
    "ui",
    "utils",
    "visualization",
]


def __getattr__(name: str):
    """Lazily import top-level src subpackages to avoid import-time side effects."""
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
