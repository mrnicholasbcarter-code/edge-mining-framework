import operator
from typing import Dict, Any, Callable

OPERATORS: Dict[str, Callable] = {
    "==": operator.eq, "!=": operator.ne, ">": operator.gt,
    "<": operator.lt, ">=": operator.ge, "<=": operator.le,
}

class FeatureEvaluator:
    @staticmethod
    def evaluate_rule(feature_value: float | str, op_str: str, threshold: Any) -> bool:
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
        for rule in rule_config:
            feat_name = rule["feature"]
            if feat_name not in features:
                return False
            val = features[feat_name]
            if not cls.evaluate_rule(val, rule["operator"], rule["threshold"]):
                return False
        return True
