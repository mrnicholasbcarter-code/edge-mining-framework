import pytest
from edge_mining_framework.gate import ExpectedValueGate

def test_breakeven_ev():
    # 50% chance of winning, costs 50 cents, payout 100
    # Gross profit = 50. Fee = 50 * 0.07 = 3.5 cents. Net = 46.5
    # EV = (0.5 * 46.5) + (0.5 * -50) = 23.25 - 25 = -1.75
    # Meaning EV is negative due to fees!
    is_profitable = ExpectedValueGate.is_profitable_after_fees(
        predicted_win_prob=0.5,
        current_contract_price_cents=50,
        exchange_fee_pct=0.07,
        minimum_ev_cents=0.0
    )
    assert is_profitable is False

def test_profitable_arb():
    # 90% chance of winning, costs 10 cents.
    is_profitable = ExpectedValueGate.is_profitable_after_fees(
        predicted_win_prob=0.9,
        current_contract_price_cents=10,
        exchange_fee_pct=0.07,
        minimum_ev_cents=10.0
    )
    assert is_profitable is True
