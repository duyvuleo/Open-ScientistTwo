"""Versioned, evidence-scoped MLAgentBench testbed integrations.

Registration describes support; readiness requires separately recorded execution
evidence on actual task data. Importing this package never contacts a model.
"""

from .catalog import get_recipe, recipes

__all__ = ["get_recipe", "recipes"]
