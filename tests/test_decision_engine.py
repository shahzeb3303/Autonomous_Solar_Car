"""
Unit Tests for Decision Engine.

Tests cover:
- Model loading
- Inference functionality
- Confidence thresholds
- Action prediction
- Performance metrics

Run with: pytest tests/test_decision_engine.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.decision.decision_engine import DecisionEngine
from src.decision.rule_engine import RuleEngine, Actions, HybridDecisionMaker
from src.sensors.sensor_fusion import FusedSensorData


class TestDecisionEngine:
    """Tests for the ML decision engine."""

    def test_engine_creation(self):
        """Test decision engine creation."""
        engine = DecisionEngine()

        assert not engine.is_loaded
        assert engine.confidence_threshold == 0.7

    def test_engine_loads_model(self):
        """Test model loading (mock mode if no model file)."""
        engine = DecisionEngine()
        result = engine.load_model()

        assert result is True
        assert engine.is_loaded

    def test_prediction_returns_valid_action(self):
        """Test that prediction returns valid action ID."""
        engine = DecisionEngine()
        engine.load_model()

        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=100,
            front_min_distance=100,
            is_valid=True,
        )

        action, confidence = engine.predict(data)

        assert 0 <= action <= 6
        assert 0.0 <= confidence <= 1.0

    def test_prediction_confidence_threshold(self):
        """Test confidence threshold checking."""
        engine = DecisionEngine(confidence_threshold=0.8)
        engine.load_model()

        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=100,
            front_min_distance=100,
            is_valid=True,
        )

        action, confidence = engine.predict(data)

        is_confident = engine.is_confident(confidence)
        assert isinstance(is_confident, bool)

    def test_action_name_mapping(self):
        """Test action ID to name mapping."""
        engine = DecisionEngine()

        assert engine.get_action_name(0) == "FORWARD"
        assert engine.get_action_name(4) == "STOP"
        assert engine.get_action_name(99) == "UNKNOWN(99)"

    def test_statistics_tracking(self):
        """Test that statistics are tracked."""
        engine = DecisionEngine()
        engine.load_model()

        data = FusedSensorData(front_min_distance=100, is_valid=True)

        # Make some predictions
        for _ in range(5):
            engine.predict(data)

        stats = engine.get_statistics()

        assert stats["predictions"] == 5
        assert "avg_inference_ms" in stats


class TestRuleEngine:
    """Tests for the rule-based decision engine."""

    def test_rule_engine_creation(self):
        """Test rule engine creation."""
        engine = RuleEngine()
        assert engine is not None

    def test_critical_distance_returns_stop(self):
        """Test that critical distance triggers STOP."""
        engine = RuleEngine()

        data = FusedSensorData(
            front_distance=15,
            front_wp_distance=15,
            front_min_distance=15,
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.STOP
        assert confidence == 1.0

    def test_clear_zone_returns_forward(self):
        """Test that clear zone triggers FORWARD."""
        engine = RuleEngine()

        data = FusedSensorData(
            front_distance=200,
            front_wp_distance=200,
            front_min_distance=200,
            left_distance=100,
            right_distance=100,
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.FORWARD

    def test_danger_zone_with_left_clear(self):
        """Test danger zone turns left when left is clear."""
        engine = RuleEngine()

        data = FusedSensorData(
            front_distance=35,
            front_wp_distance=35,
            front_min_distance=35,
            left_distance=100,  # Clear
            right_distance=30,  # Blocked
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.TURN_LEFT

    def test_danger_zone_with_right_clear(self):
        """Test danger zone turns right when right is clear."""
        engine = RuleEngine()

        data = FusedSensorData(
            front_distance=35,
            front_wp_distance=35,
            front_min_distance=35,
            left_distance=30,   # Blocked
            right_distance=100, # Clear
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.TURN_RIGHT

    def test_caution_zone_returns_slow_down(self):
        """Test that caution zone triggers SLOW_DOWN."""
        engine = RuleEngine()

        data = FusedSensorData(
            front_distance=75,
            front_wp_distance=75,
            front_min_distance=75,
            left_distance=100,
            right_distance=100,
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.SLOW_DOWN

    def test_person_detected_returns_stop(self):
        """Test that person detection triggers STOP."""
        engine = RuleEngine()

        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=100,
            front_min_distance=100,
            camera_object_detected=1,
            camera_object_class=1,  # Person
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.STOP

    def test_side_proximity_turns_away(self):
        """Test that close side obstacle triggers turn away."""
        engine = RuleEngine()

        # Left side close
        data = FusedSensorData(
            front_distance=150,
            front_wp_distance=150,
            front_min_distance=150,
            left_distance=20,  # Too close
            right_distance=100,
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.TURN_RIGHT

    def test_heading_error_correction(self):
        """Test that heading error triggers correction."""
        engine = RuleEngine()

        # Heading error to the right
        data = FusedSensorData(
            front_distance=200,
            front_wp_distance=200,
            front_min_distance=200,
            left_distance=100,
            right_distance=100,
            heading_error=25,  # Need to turn right
            is_valid=True,
        )

        action, confidence = engine.decide(data)

        assert action == Actions.TURN_RIGHT


class TestHybridDecisionMaker:
    """Tests for the hybrid ML/rule decision maker."""

    def test_hybrid_creation(self):
        """Test hybrid decision maker creation."""
        ml_engine = DecisionEngine()
        ml_engine.load_model()
        rule_engine = RuleEngine()

        hybrid = HybridDecisionMaker(
            decision_engine=ml_engine,
            rule_engine=rule_engine,
            confidence_threshold=0.7,
        )

        assert hybrid is not None

    def test_hybrid_returns_valid_action(self):
        """Test that hybrid returns valid action."""
        ml_engine = DecisionEngine()
        ml_engine.load_model()
        rule_engine = RuleEngine()

        hybrid = HybridDecisionMaker(ml_engine, rule_engine)

        data = FusedSensorData(front_min_distance=100, is_valid=True)

        action, confidence, source = hybrid.decide(data)

        assert 0 <= action <= 6
        assert source in ("ML", "RULES")

    def test_hybrid_statistics(self):
        """Test hybrid statistics tracking."""
        ml_engine = DecisionEngine()
        ml_engine.load_model()
        rule_engine = RuleEngine()

        hybrid = HybridDecisionMaker(ml_engine, rule_engine)

        data = FusedSensorData(front_min_distance=100, is_valid=True)

        for _ in range(10):
            hybrid.decide(data)

        stats = hybrid.get_statistics()

        assert stats["ml_decisions"] + stats["rule_decisions"] == 10


class TestRuleValidation:
    """Tests for rule engine validation against expected behavior."""

    def test_validate_ml_agreement(self):
        """Test ML vs rules validation."""
        ml_engine = DecisionEngine()
        ml_engine.load_model()
        rule_engine = RuleEngine()

        data = FusedSensorData(
            front_distance=100,
            front_wp_distance=100,
            front_min_distance=100,
            is_valid=True,
        )

        ml_action, ml_conf = ml_engine.predict(data)

        validation = rule_engine.validate_against_ml(data, ml_action, ml_conf)

        assert "ml_action" in validation
        assert "rule_action" in validation
        assert "agreement" in validation
