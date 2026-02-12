"""
Core Module for Autonomous Solar Vehicle.

This module provides the main runtime components:
- MainLoop: Primary autonomy loop (sense → perceive → decide → act)
- StateMachine: Vehicle state management
- EventBus: Internal pub/sub for component communication

Entry Point:
    python -m src.core.main_loop
"""

from src.core.state_machine import StateMachine, VehicleState
from src.core.main_loop import AutonomousVehicle

__all__ = [
    "AutonomousVehicle",
    "StateMachine",
    "VehicleState",
]
