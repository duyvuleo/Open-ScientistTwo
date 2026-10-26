"""Compatibility import; new and recorded live paths share the research engine.

Historical execution still requires its original frozen implementation identity.
"""
from ..research.controller import ResearchController, AgentOutputError, ALLOWED, PROMPTS

LiveController = ResearchController

__all__ = ["LiveController", "ResearchController", "AgentOutputError", "ALLOWED", "PROMPTS"]
