"""Tuning-free high-resolution generation techniques.

Each technique module exposes a common interface so that the generation
engine can apply any technique interchangeably.
"""

from rsgen8k.techniques.registry import get_technique, list_techniques, TECHNIQUE_REGISTRY

__all__ = ["get_technique", "list_techniques", "TECHNIQUE_REGISTRY"]
