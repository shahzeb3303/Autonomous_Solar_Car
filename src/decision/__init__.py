"""
Decision Module for Autonomous Solar Vehicle.

This module provides the decision-making pipeline:
- SafetyGovernor: Hard safety overrides (Layer 3)
- DecisionEngine: ML-based decision inference (Layer 2)
- RuleEngine: Rule-based fallback decisions (Layer 1)
- ActionExecutor: Translates actions to motor commands

Safety Hierarchy:
    Layer 3 (Outermost): SAFETY GOVERNOR - hard-coded rules, NOT ML
        → front < 20cm = STOP (no ML override possible)
        → person detected = STOP (no ML override possible)
        → sensor error = STOP

    Layer 2 (Middle): ML DECISION ENGINE - trained model inference
        → Takes fused sensor data, predicts best action
        → Operates only when Layer 3 hasn't triggered STOP

    Layer 1 (Innermost): RULE ENGINE - fallback if ML fails
        → Simple if/else rules matching training data rules
        → Activates if ML confidence < 0.7 or model load fails
"""

from src.decision.safety_governor import SafetyGovernor, SafetyStatus
from src.decision.decision_engine import DecisionEngine
from src.decision.rule_engine import RuleEngine
from src.decision.action_executor import ActionExecutor, Action

__all__ = [
    "SafetyGovernor",
    "SafetyStatus",
    "DecisionEngine",
    "RuleEngine",
    "ActionExecutor",
    "Action",
]
