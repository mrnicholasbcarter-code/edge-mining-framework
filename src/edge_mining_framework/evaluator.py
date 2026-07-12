"""
Feature Evaluator Module

Provides the FeatureEvaluator class for evaluating compound feature rules
against feature dictionaries. Uses Python's operator module for O(1) safe
operator dispatch, avoiding slow and unsafe eval() operations.

Features:
    - Compound rule evaluation with fail-fast logic (short-circuit on first failure)
    - O(1) operator dispatch via operator module
    - Agnostic to data source - evaluates raw dictionaries only
    - No exchange SDK dependencies

Supported Operators:
    == (eq), != (ne), > (gt), < (lt), >= (ge), <= (le)
    in_range: threshold is [min, max], returns min <= value <= max
    in_set: threshold is iterable, returns value in threshold

Example:
    >>> from edge_mining_framework import FeatureEvaluator
    >>> features = {"rsi": 30.5, "volume_spike": True}
    >>> rules = [
    ...     {"feature": "rsi", "operator": "<", "threshold": 40.0},
    ...     {"feature": "volume_spike", "operator": "==", "threshold": True}
    ... ]
    >>> FeatureEvaluator.evaluate_compound(features, rules)
    True

Architecture:
    FeatureEvaluator is strictly agnostic to data sources. Do not import
    KalshiClient or any exchange SDKs into this module. It evaluates raw
    dictionaries only.
"""

import operator
from typing import Dict, Any, Callable


# O(1) operator dispatch table - avoids eval() for safety and performance
OPERATORS: Dict[str, Callable] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


class FeatureEvaluator:
    """
    Evaluates compound feature rules against feature dictionaries.
    
    Uses fail-fast logic: returns False immediately on first failed condition on the first failed
    rule to preserve CPU cycles on downstream feature resolution.
    
    No exchange SDK imports allowed - evaluates raw dictionaries only.
    """
    
    @staticmethod
    def evaluate_rule(feature_value: float | str, op_str: str, threshold: Any) -> bool:
        """
        Evaluate a single feature rule against a threshold.
        
        Args:
            feature_value: The value of the feature to evaluate (numeric or string)
            op_str: Operator string - one of ==, !=, >, <, >=, <=, in_range, in_set
            threshold: Threshold value - scalar for comparisons, [min, max] for in_range,
                      iterable for in_set
            
        Returns:
            bool: True if rule passes, False otherwise
            
        Raises:
            ValueError: If op_str is not a supported operator
            
        Examples:
            >>> FeatureEvaluator.evaluate_rule(5, "==", 5)
            True
            >>> FeatureEvaluator.evaluate_rule(5, "in_range", [1, 10])
            True
            >>> FeatureEvaluator.evaluate_rule("a", "in_set", ["a", "b", "c"])
            True
        """
        if op_str in OPERATORS:
            return OPERATORS[op_str](feature_value, threshold)
        elif op_str == "in_range":
            return threshold[0] <= feature_value <= threshold[1]
        elif op_str == "in_set":
            return feature_value in threshold
        else:
            raise ValueError(f"Unknown operator: {op_str}")

    @classmethod
    def evaluate_compound(cls, features: Dict[str, Any], rule_config: list[dict]) -> bool:
        """
        Evaluate multiple rules against features with fail-fast short-circuit.
        
        Iterates through rules in order. Returns False immediately on first
        failure. Missing features return False (not an error).
        
        Args:
            features: Dictionary mapping feature names to values
            rule_config: List of rule dictionaries, each with keys:
                - "feature": Feature name (str)
                - "operator": Operator string (str)
                - "threshold": Threshold value (Any)
                
        Returns:
            bool: True if all rules pass, False if any rule fails or feature missing
            
        Raises:
            KeyError: If rule dictionary missing required keys (operator, threshold)
            ValueError: If operator is not supported
            
        Examples:
            >>> features = {"rsi": 30.5, "volume_spike": True}
            >>> rules = [
            ...     {"feature": "rsi", "operator": "<", "threshold": 40.0},
            ...     {"feature": "volume_spike", "operator": "==", "threshold": True}
            ... ]
            >>> FeatureEvaluator.evaluate_compound(features, rules)
            True
            
            >>> features = {"rsi": 50.0}
            >>> FeatureEvaluator.evaluate_compound(features, rules)
            False
            
            >>> FeatureEvaluator.evaluate_compound({"x": 1}, [])
            True
        """
        for rule in rule_config:
            feat_name = rule["feature"]
            if feat_name not in features:
                return False
            val = features[feat_name]
            if not cls.evaluate_rule(val, rule["operator"], rule["threshold"]):
                return False
        return True