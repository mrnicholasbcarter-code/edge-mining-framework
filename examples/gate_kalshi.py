#!/usr/bin/env python3
"""
Kalshi-specific EV gate demonstration.

Prints continuous EV and Kelly fraction for both negative-EV and positive-EV
trades, showing when the gate should permit or reject orders.

Usage:
    python -m examples.gate_kalshi
"""

from edge_mining_framework import ExpectedValueGate


def main() -> None:
    """Run two Kalshi trade scenarios through the EV gate."""
    # Kalshi fee = 4 cents per $1 contract (typical as of 2024)
    KALSHI_FEE = 0.04
    PAYOUT = 100  # cents per contract (fixed)

    # NEGATIVE-EV trade (should be REJECTED)
    # 48% win probability implies the YES side is overpriced; EV is negative.
    neg_ev = ExpectedValueGate.calculate_ev_metrics(
        predicted_win_prob=0.48,
        current_contract_price_cents=48,  # priced at ~48c on a fair 48% outcome
        payout_cents=PAYOUT,
        exchange_fee_pct=KALSHI_FEE,
        bankroll=500.0,
    )

    print("=" * 60)
    print("GATE REJECTS: 48% win prob @ 48c with 4% fee")
    print("=" * 60)
    print(f"  EV (cents):        {neg_ev.expected_value:.2f}")
    print(f"  Kelly fraction:    {neg_ev.kelly_fraction:.4f}")
    print(f"  Recommended size:  ${neg_ev.recommended_size:.2f}")
    print(f"  Profitable:        {neg_ev.profitable}")
    print()

    # POSITIVE-EV trade (should be ACCEPTED)
    # 57% win probability with better-than-priced edge.
    pos_ev = ExpectedValueGate.calculate_ev_metrics(
        predicted_win_prob=0.57,
        current_contract_price_cents=48,  # market is underpricing this outcome
        payout_cents=PAYOUT,
        exchange_fee_pct=KALSHI_FEE,
        bankroll=500.0,
    )

    print("=" * 60)
    print("GATE ACCEPTS: 57% win prob @ 48c with 4% fee")
    print("=" * 60)
    print(f"  EV (cents):        {pos_ev.expected_value:.2f}")
    print(f"  Kelly fraction:    {pos_ev.kelly_fraction:.4f}")
    print(f"  Recommended size:  ${pos_ev.recommended_size:.2f}")
    print(f"  Profitable:        {pos_ev.profitable}")


if __name__ == "__main__":
    main()