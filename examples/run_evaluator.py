#!/usr/bin/env python3
"""
Run all rules from examples/rules.yaml with sample feature data and print
EV + Kelly-fraction outputs.

Usage:
    python -m examples.run_evaluator
"""

from __future__ import annotations

from pathlib import Path

import yaml

from edge_mining_framework import ExpectedValueGate, FeatureEvaluator

# Sample feature data for live inspection. The series are intentionally
# varied to demonstrate scalar and series operators.
SAMPLE_FEATURES = {
    # --- Scalar features -------------------------------------------------------
    "rsi": 28,
    "vol_expansion": True,
    "volume_spike": True,
    "implied_prob": 0.85,
    # --- Series features (for series operators) -------------------------------
    "series": {
        # price crosses above 50 in rule 2
        "price": [45, 47, 48, 49, 51, 52],
        # momentum z-score for rule 3  (last value deviates > 2 std)
        "momentum_z": [0.1, 0.2, -0.1, 0.3, 2.2],
        # simulated returns for rule 4
        "market_ret": [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.02],
        "sister_ret": [0.012, -0.018, 0.028, -0.012, 0.018, 0.038, -0.019],
        # implied probability history for rule 5
        "implied_prob_series": [0.45, 0.48, 0.52, 0.55, 0.62, 0.70, 0.78, 0.85],
    },
}


def main() -> None:
    """Load rules.yaml, evaluate each rule against sample features, and print EV + Kelly."""
    rules_path = Path(__file__).parent / "rules.yaml"
    with open(rules_path) as fh:
        config = yaml.safe_load(fh)

    rules = config["rules"]

    print("=" * 72)
    print("EDGE-MINING-FRAMEWORK — Rule Evaluation + EV + Kelly Sizing Report")
    print("=" * 72)

    for rule in rules:
        name = rule["name"]
        description = rule["description"]
        signal_rules = rule["signal"]
        trade = rule["trade"]

        # Evaluate the signal conditions
        passed = FeatureEvaluator.evaluate_compound(SAMPLE_FEATURES, signal_rules)

        # Compute EV + Kelly and print results
        ev = ExpectedValueGate.calculate_ev_metrics(
            predicted_win_prob=trade["win_prob"],
            current_contract_price_cents=trade["price_cents"],
            payout_cents=trade["payout_cents"],
            exchange_fee_pct=trade["fee_pct"],
            bankroll=trade["bankroll"],
        )

        print(f"\n--- Rule: {name} ---\n")
        print(f"  Signal: {description}")
        print(f"  Conditions met: {passed}")
        print(
            f"  Trade params: win_prob={trade['win_prob']}, price={trade['price_cents']}c, "
            f"fee={trade['fee_pct'] * 100}%"
        )
        print(f"  EV (cents/trade):     {ev.expected_value:.2f}")
        print(f"  Kelly fraction:       {ev.kelly_fraction:.4f}")
        print(f"  Recommended size:       ${ev.recommended_size:.2f}")
        print(f"  Profitable (>0 EV):   {ev.profitable}")


if __name__ == "__main__":
    main()
