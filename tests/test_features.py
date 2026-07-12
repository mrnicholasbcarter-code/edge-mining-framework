import pytest
from edge_mining_framework.evaluator import FeatureEvaluator


class TestEvaluateRule:
    def test_equals(self):
        assert FeatureEvaluator.evaluate_rule(5, "==", 5) is True

    def test_not_equals(self):
        assert FeatureEvaluator.evaluate_rule(5, "!=", 3) is True

    def test_greater_than(self):
        assert FeatureEvaluator.evaluate_rule(10, ">", 5) is True
        assert FeatureEvaluator.evaluate_rule(5, ">", 10) is False

    def test_less_than(self):
        assert FeatureEvaluator.evaluate_rule(3, "<", 10) is True

    def test_gte(self):
        assert FeatureEvaluator.evaluate_rule(5, ">=", 5) is True
        assert FeatureEvaluator.evaluate_rule(4, ">=", 5) is False

    def test_lte(self):
        assert FeatureEvaluator.evaluate_rule(5, "<=", 5) is True

    def test_in_range(self):
        assert FeatureEvaluator.evaluate_rule(5, "in_range", [1, 10]) is True
        assert FeatureEvaluator.evaluate_rule(15, "in_range", [1, 10]) is False

    def test_in_set(self):
        assert FeatureEvaluator.evaluate_rule("a", "in_set", ["a", "b", "c"]) is True
        assert FeatureEvaluator.evaluate_rule("z", "in_set", ["a", "b"]) is False

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            FeatureEvaluator.evaluate_rule(5, "~~", 5)


class TestEvaluateCompound:
    def test_all_rules_pass(self):
        features = {"rsi": 30.5, "volume_spike": True}
        rules = [
            {"feature": "rsi", "operator": "<", "threshold": 40.0},
            {"feature": "volume_spike", "operator": "==", "threshold": True}
        ]
        assert FeatureEvaluator.evaluate_compound(features, rules) is True

    def test_one_rule_fails(self):
        features = {"rsi": 50.0, "volume_spike": True}
        rules = [
            {"feature": "rsi", "operator": "<", "threshold": 40.0},
            {"feature": "volume_spike", "operator": "==", "threshold": True}
        ]
        assert FeatureEvaluator.evaluate_compound(features, rules) is False

    def test_missing_feature_returns_false(self):
        features = {"rsi": 30}
        rules = [{"feature": "missing_feat", "operator": "==", "threshold": True}]
        assert FeatureEvaluator.evaluate_compound(features, rules) is False

    def test_empty_rules_returns_true(self):
        assert FeatureEvaluator.evaluate_compound({"x": 1}, []) is True

    def test_empty_rule_dictionary_raises_keyerror(self):
        features = {"rsi": 30}
        rules = [{}]
        with pytest.raises(KeyError, match="feature"):
            FeatureEvaluator.evaluate_compound(features, rules)
            
    def test_invalid_operator_in_compound(self):
        features = {"rsi": 30.5}
        rules = [{"feature": "rsi", "operator": "INVALID_OP", "threshold": 40.0}]
        with pytest.raises(ValueError, match="Unknown operator"):
            FeatureEvaluator.evaluate_compound(features, rules)

    def test_missing_operator_or_threshold_raises_keyerror(self):
        features = {"rsi": 30.5}
        rules = [{"feature": "rsi", "threshold": 40.0}]
        with pytest.raises(KeyError, match="operator"):
            FeatureEvaluator.evaluate_compound(features, rules)
