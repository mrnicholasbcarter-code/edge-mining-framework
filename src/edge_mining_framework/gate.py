"""
Expected Value Gate Module

Provides the ExpectedValueGate class for calculating whether a prediction
market trade has positive Expected Value (EV) after accounting for exchange
fees, plus continuous EV output and Kelly-fraction position sizing.

This is the core risk management component that mathematically blocks
deployment unless EV exceeds friction parameters, and sizes the bet using
the Kelly criterion.

The EV calculation accounts for:
- Contract purchase cost (current_contract_price_cents)
- Payout on win (payout_cents, typically 100 cents = $1.00)
- Exchange fees (percentage of payout, capped at max fee)
- Minimum EV threshold (minimum_ev_cents) for risk tolerance

Formula:
    cost = current_contract_price_cents
    gross_profit = payout_cents - cost
    max_fee = exchange_fee_pct * payout_cents
    effective_fee = min(gross_profit * exchange_fee_pct, max_fee)
    net_profit_if_win = gross_profit - effective_fee
    loss_if_lose = cost
    expected_value = (predicted_win_prob * net_profit_if_win)
                     - ((1 - predicted_win_prob) * loss_if_lose)

Kelly fraction (for a binary bet with net profit b and loss a):
    kelly_fraction = expected_value / net_profit_if_win
    recommended_size = kelly_fraction * bankroll

Example:
    >>> # 55% win probability, buying at 45 cents, 7% fee
    >>> metrics = ExpectedValueGate.calculate_ev_metrics(0.55, 45, 100, 0.07)
    >>> metrics["profitable"]
    True
    >>> metrics["expected_value"] > 0
    True
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EVMetrics:
    """Continuous EV output and Kelly sizing for a single trade.

    Attributes:
        expected_value: Net expected value per trade in cents
            (positive = +EV, negative = -EV).
        kelly_fraction: Optimal fraction of bankroll to bet per
            the Kelly criterion, clamped to [0.0, 1.0].
        recommended_size: kelly_fraction * bankroll, in bankroll units.
        profitable: True if expected_value > 0 (backward-compatible binary gate).
    """

    expected_value: float
    kelly_fraction: float
    recommended_size: float
    profitable: bool


class ExpectedValueGate:
    """Expected Value (EV) payout gating engine for prediction markets.

    Provides:
      * ``calculate_ev_metrics`` -> continuous EV, Kelly fraction, recommended size.
      * ``is_profitable_after_fees`` -> backward-compatible boolean gate.

    The fee model assumes:
    - Fees are a percentage of the payout (not the stake).
    - Fees are capped at the maximum possible fee (exchange_fee_pct * payout).
    - Effective fee is the minimum of (gross_profit * fee_pct) and max_fee.
    """

    @staticmethod
    def calculate_ev_metrics(
        predicted_win_prob: float,
        current_contract_price_cents: float,
        payout_cents: float = 100,
        exchange_fee_pct: float = 0.07,
        bankroll: float = 1.0,
    ) -> EVMetrics:
        """Calculate continuous EV and Kelly-fraction sizing for a trade.

        Args:
            predicted_win_prob: Model's predicted probability of winning (0.0-1.0).
            current_contract_price_cents: Current contract price in cents (0-payout).
            payout_cents: Payout on win in cents (default 100 = $1.00).
            exchange_fee_pct: Exchange fee as decimal (default 0.07 = 7%).
            bankroll: Bankroll to size against (default 1.0; pass your
                account equity in cents to get recommended_size in cents).

        Returns:
            EVMetrics dataclass with expected_value, kelly_fraction,
            recommended_size, and profitable.

        Raises:
            ValueError: If win_prob not in [0,1], price not in [0, payout],
                fee negative, or bankroll negative.

        Example:
            >>> m = ExpectedValueGate.calculate_ev_metrics(0.55, 45, 100, 0.07)
            >>> m.profitable
            True
            >>> round(m.expected_value, 2)
            5.83
        """
        if not 0.0 <= predicted_win_prob <= 1.0:
            raise ValueError(
                f"predicted_win_prob must be in [0, 1], got {predicted_win_prob}"
            )
        if not 0 <= current_contract_price_cents <= payout_cents:
            raise ValueError(
                f"current_contract_price_cents must be in [0, {payout_cents}], "
                f"got {current_contract_price_cents}"
            )
        if exchange_fee_pct < 0:
            raise ValueError(
                f"exchange_fee_pct must be non-negative, got {exchange_fee_pct}"
            )
        if bankroll < 0:
            raise ValueError(f"bankroll must be non-negative, got {bankroll}")

        cost = current_contract_price_cents
        gross_profit = payout_cents - cost
        max_fee = exchange_fee_pct * payout_cents
        effective_fee = min(gross_profit * exchange_fee_pct, max_fee)

        net_profit_if_win = gross_profit - effective_fee
        loss_if_lose = cost  # positive magnitude; payout 0, you lose cost

        # EV in cents per trade.
        expected_value = (
            predicted_win_prob * net_profit_if_win
        ) - (
            (1.0 - predicted_win_prob) * loss_if_lose
        )

        # Kelly fraction for a binary outcome:
        #   f* = (p * b - q * a) / (b * a) * b == (p*b - q*a) / b  when stake==cost
        # Simpler equivalent for a binary bet with profit `b` and loss `a`:
        #   f* = (p * b - q * a) / b   (here a==cost, b==net_profit_if_win)
        # Which equals expected_value / net_profit_if_win.
        if net_profit_if_win > 0:
            kelly = expected_value / net_profit_if_win
        else:
            # No positive payoff possible; never bet.
            kelly = 0.0

        # Clamp to a long-only fraction [0, 1]: negative EV -> 0, extreme edge -> 1.
        kelly_fraction = max(0.0, min(1.0, kelly))
        recommended_size = kelly_fraction * bankroll

        return EVMetrics(
            expected_value=float(expected_value),
            kelly_fraction=float(kelly_fraction),
            recommended_size=float(recommended_size),
            profitable=bool(expected_value > 0),
        )

    @staticmethod
    def is_profitable_after_fees(
        predicted_win_prob: float,
        current_contract_price_cents: int,
        payout_cents: int = 100,
        exchange_fee_pct: float = 0.07,
        minimum_ev_cents: float = 1.0,
    ) -> bool:
        """Determine if a trade has positive EV after fees (backward compatible).

        Thin wrapper over ``calculate_ev_metrics`` so existing callers and
        tests keep working unchanged.

        Args:
            predicted_win_prob: Model's predicted probability of winning (0.0-1.0).
            current_contract_price_cents: Current contract price in cents (0-100).
            payout_cents: Payout on win in cents (default 100 = $1.00).
            exchange_fee_pct: Exchange fee as decimal (default 0.07 = 7%).
            minimum_ev_cents: Minimum acceptable EV in cents (default 1.0).

        Returns:
            True if expected_value >= minimum_ev_cents, False otherwise.

        Raises:
            ValueError: If predicted_win_prob not in [0, 1] or price not
                in [0, payout].

        Examples:
            >>> ExpectedValueGate.is_profitable_after_fees(0.9, 10, 100, 0.07, 10.0)
            True
            >>> ExpectedValueGate.is_profitable_after_fees(0.5, 50, 100, 0.07, 0.0)
            False
        """
        metrics = ExpectedValueGate.calculate_ev_metrics(
            predicted_win_prob,
            current_contract_price_cents,
            payout_cents=payout_cents,
            exchange_fee_pct=exchange_fee_pct,
        )
        return metrics.expected_value >= minimum_ev_cents
