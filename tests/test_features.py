import pytest
from edge_mining_framework.evaluator import FeatureEvaluator

def test_compound_evaluation():
    features = {"rsi": 30.5, "volume_spike": True}
    
    rules = [
        {"feature": "rsi", "operator": "<", "threshold": 40.0},
        {"feature": "volume_spike", "operator": "==", "threshold": True}
    ]
    
    assert FeatureEvaluator.evaluate_compound(features, rules) is True
