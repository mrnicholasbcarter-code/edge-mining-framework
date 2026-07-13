"""
Feature Evaluator Module

Provides the FeatureEvaluator class for evaluating compound feature rules
against feature dictionaries. Uses Python's operator module for O(1) safe
scalar operator dispatch, and numpy for the time-series operators
(`crosses_above`, `crosses_below`, `zscore`, `rolling_corr`, `rank`),
avoiding slow and unsafe eval() operations.

Features:
    - Compound rule evaluation with fail-fast logic (short-circuit on first failure)
    - O(1) scalar operator dispatch via operator module
    - O(n) series operators via numpy (no eval())
    - Agnostic to data source - evaluates raw dictionaries only
    - No exchange SDK dependencies

Supported Scalar Operators:
    == (eq), != (ne), > (gt), < (lt), >= (ge), <= (le)
    in_range: threshold is [min, max], returns min <= value <= max
    in_set: threshold is iterable, returns value in threshold

Supported Series Operators (require a list of recent values under
``features["series"]["<series_name>"]`` or a directly-passed list):
    crosses_above: threshold scalar -> True if last value <= threshold
        and prior value > threshold is WRONG; actually True if the most
        recent value crossed from <= threshold to > threshold (i.e.
        series[-2] <= threshold < series[-1]).
    crosses_below: True if series[-2] >= threshold > series[-1].
    zscore: threshold scalar -> True if z-score of last value vs the
        series exceeds threshold (|z| > threshold). Use to detect
        outliers / statistically significant deviations.
    rolling_corr: threshold (dict with ``window`` and ``other`` naming a
        second series in features["series"]) -> True if abs rolling
        correlation of the last `window` values between the two series
        >= threshold.
    rank: threshold scalar -> True if the rank (percentile, 0-1) of the
        last value within the series is >= threshold. Useful for "value
        is in the top X% of its recent history".

Operator interface accepts either:
  - a scalar value (for scalar operators), or
  - a list of recent values (for series operators). When a series
    operator is used, the rule's ``feature`` key names a series. The
    series itself may be supplied as:
      * features[feat_name] == list  (inline series), or
      * features["series"][feat_name] == list  (namespaced series).

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
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# O(1) operator dispatch table - avoids eval() for safety and performance
OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}

SERIES_OPERATORS = {"crosses_above", "crosses_below", "zscore", "rolling_corr", "rank"}


def _resolve_series(features: dict[str, Any], feat_name: str) -> np.ndarray:
    """Resolve a series value for a series operator.

    Accepts either an inline list under features[feat_name] or a namespaced
    list under features["series"][feat_name].

    Returns a 1-D float ndarray. Returns an empty array if missing/empty.

    Args:
        features: The full feature dictionary.
        feat_name: The feature (series) name.

    Returns:
        np.ndarray of floats (possibly empty).
    """
    if feat_name in features and isinstance(features[feat_name], (list, tuple, np.ndarray)):
        return np.asarray(features[feat_name], dtype=float)
    series_block = features.get("series", {})
    if isinstance(series_block, dict) and feat_name in series_block:
        return np.asarray(series_block[feat_name], dtype=float)
    return np.asarray([], dtype=float)


def _op_crosses_above(series: np.ndarray, threshold: float) -> bool:
    """True if series[-2] <= threshold < series[-1] (upward cross)."""
    if series.size < 2:
        return False
    return bool(series[-2] <= threshold < series[-1])


def _op_crosses_below(series: np.ndarray, threshold: float) -> bool:
    """True if series[-2] >= threshold > series[-1] (downward cross)."""
    if series.size < 2:
        return False
    return bool(series[-2] >= threshold > series[-1])


def _op_zscore(series: np.ndarray, threshold: float) -> bool:
    """True if |z(last)| >= threshold, where z = (last - mean) / std."""
    if series.size < 2:
        return False
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=0))
    if std == 0.0:
        return False
    z = abs(float(series[-1]) - mean) / std
    return bool(z >= threshold)


def _op_rolling_corr(series: np.ndarray, threshold: float, other: np.ndarray) -> bool:
    """True if |corr(last `window` of series, last `window` of other)| >=
    threshold. ``window`` is read from the threshold dict; threshold itself
    is the minimum |corr| gate.
    """
    # threshold is a dict: {"other": "<name>", "window": int, "min_corr": float}
    if not isinstance(threshold, dict):
        raise ValueError(
            "rolling_corr threshold must be a dict with 'other', 'window', and 'min_corr'"
        )
    window = int(threshold["window"])
    min_corr = float(threshold["min_corr"])
    if series.size < window or other.size < window:
        return False
    a = series[-window:]
    b = other[-window:]
    if a.std() == 0 or b.std() == 0:
        return False
    corr = float(np.corrcoef(a, b)[0, 1])
    return bool(abs(corr) >= min_corr)


def _op_rank(series: np.ndarray, threshold: float) -> bool:
    """True if the percentile rank (0..1) of the last value within the
    series is >= threshold. Rank = (# values <= last) / (len - 1)
    (tie-aware average rank via scipy not assumed; we use simple ordinal-ish rank).
    """
    if series.size < 2:
        return False
    last = float(series[-1])
    # Average rank: fraction of strictly-less values plus half of ties.
    strictly_less = float(np.sum(series < last))
    ties = float(np.sum(series == last))
    n = series.size
    # average rank in [0, 1]
    rank = (strictly_less + ties / 2.0) / (n - 1) if n > 1 else 0.5
    return bool(rank >= threshold)


class FeatureEvaluator:
    """Evaluates compound feature rules against feature dictionaries.

    Uses fail-fast logic: returns False immediately on first failed
    condition on the first failed rule to preserve CPU cycles on
    downstream feature resolution.

    Operator interface accepts either a scalar (for scalar operators) or
    a list of recent values (for series operators). Series operators are
    dispatched via numpy, never ``eval()``.

    No exchange SDK imports allowed - evaluates raw dictionaries only.
    """

    @staticmethod
    def evaluate_rule(
        feature_value: float | str | Sequence[Any] | np.ndarray[Any, Any],
        op_str: str,
        threshold: Any,
        features: dict[str, Any] | None = None,
        feat_name: str | None = None,
    ) -> bool:
        """Evaluate a single feature rule against a threshold.

        Args:
            feature_value: The value of the feature (scalar, string, or a
                list/sequence for series operators).
            op_str: Operator string - one of ==, !=, >, <, >=, <=, in_range,
                in_set, crosses_above, crosses_below, zscore, rolling_corr,
                rank.
            threshold: Threshold value. Scalar for comparisons; [min, max]
                for in_range; iterable for in_set; a float gate for
                crosses_above/crosses_below/zscore/rank; a dict with
                ``other``/``window``/``min_corr`` for rolling_corr.
            features: Optional full feature dict, needed for rolling_corr to
                resolve the second (``other``) series by name.
            feat_name: Optional name of the feature (used to resolve the
                other series for rolling_corr).

        Returns:
            bool: True if rule passes, False otherwise.

        Raises:
            ValueError: If op_str is not a supported operator, or a series
                operator is invoked with an insufficient series length.

        Examples:
            >>> FeatureEvaluator.evaluate_rule(5, "==", 5)
            True
            >>> FeatureEvaluator.evaluate_rule(5, "in_range", [1, 10])
            True
            >>> FeatureEvaluator.evaluate_rule("a", "in_set", ["a", "b", "c"])
            True
            >>> # Series operator: last value crossed above 100
            >>> FeatureEvaluator.evaluate_rule([98, 99, 101], "crosses_above", 100)
            True
            >>> # zscore: last value is > 1.5 std devs from mean
            >>> FeatureEvaluator.evaluate_rule([1, 1, 1, 1, 5], "zscore", 1.5)
            True
        """
        if op_str in OPERATORS:
            return bool(OPERATORS[op_str](feature_value, threshold))
        elif op_str == "in_range":
            return bool(threshold[0] <= feature_value <= threshold[1])
        elif op_str == "in_set":
            return bool(feature_value in threshold)
        elif op_str in SERIES_OPERATORS:
            if op_str == "rolling_corr":
                if features is None or feat_name is None:
                    raise ValueError(
                        "rolling_corr requires features and feat_name to resolve the 'other' series"
                    )
                other_name = threshold["other"]
                other_series = _resolve_series(features, other_name)
                series = np.asarray(feature_value, dtype=float)
                return _op_rolling_corr(series, threshold, other_series)
            # All other series operators just need the series + scalar threshold.
            series = np.asarray(feature_value, dtype=float)
            if op_str == "crosses_above":
                return _op_crosses_above(series, float(threshold))
            elif op_str == "crosses_below":
                return _op_crosses_below(series, float(threshold))
            elif op_str == "zscore":
                return _op_zscore(series, float(threshold))
            elif op_str == "rank":
                return _op_rank(series, float(threshold))
        raise ValueError(f"Unknown operator: {op_str}")

    @classmethod
    def evaluate_compound(cls, features: dict[str, Any], rule_config: list[dict[str, Any]]) -> bool:
        """Evaluate multiple rules against features with fail-fast short-circuit.

        Iterates through rules in order. Returns False immediately on first
        failure. Missing features return False (not an error) for scalar
        operators. For series operators, a missing series also returns False
        (not an error), so callers can stream features incrementally without
        raising on warm-up.

        Args:
            features: Dictionary mapping feature names to values. May also
                contain a ``"series"`` sub-dict mapping series names to lists
                of recent values.
            rule_config: List of rule dictionaries, each with keys:
                - "feature": Feature name (str)
                - "operator": Operator string (str)
                - "threshold": Threshold value (Any)

        Returns:
            bool: True if all rules pass, False if any rule fails or
            feature/series missing.

        Raises:
            KeyError: If rule dictionary missing required keys (operator,
            threshold).
            ValueError: If operator is not supported.

        Examples:
            >>> features = {"rsi": 30.5, "volume_spike": True}
            >>> rules = [
            ...     {"feature": "rsi", "operator": "<", "threshold": 40.0},
            ...     {"feature": "volume_spike", "operator": "==", "threshold": True}
            ... ]
            >>> FeatureEvaluator.evaluate_compound(features, rules)
            True

            >>> # Series rule example
            >>> features = {"series": {"price": [10, 11, 9, 12, 13]}}
            >>> rules = [{"feature": "price", "operator": "crosses_above", "threshold": 12}]
            >>> FeatureEvaluator.evaluate_compound(features, rules)
            True
        """
        for rule in rule_config:
            feat_name = rule["feature"]
            op_str = rule["operator"]

            # Series operators resolve differently from scalars.
            if op_str in SERIES_OPERATORS:
                series = _resolve_series(features, feat_name)
                if series.size == 0:
                    return False
                # For series operators, threshold must exist
                if "threshold" not in rule:
                    raise KeyError("threshold")
                threshold = rule["threshold"]
                if not cls.evaluate_rule(
                    series, op_str, threshold, features=features, feat_name=feat_name
                ):
                    return False
                continue

            # Scalar path - check feature exists first
            if feat_name not in features:
                return False
            # Then check threshold exists
            if "threshold" not in rule:
                raise KeyError("threshold")
            threshold = rule["threshold"]
            val = features[feat_name]
            if not cls.evaluate_rule(val, op_str, threshold):
                return False
        return True
