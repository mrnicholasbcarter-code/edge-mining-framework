"""
Expected Value Gate Module

Provides the ExpectedValueGate class for calculating whether a prediction
market trade has positive Expected Value (EV) after accounting for exchange
fees. This is the core risk management component that mathematically blocks
deployment unless EV exceeds friction parameters.

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
    loss_if_lose = -cost
    expected_value = (predicted_win_prob * net_profit_if_win) + ((1 - predicted_win_prob) * loss_if_lose)
    
    Returns True if expected_value >= minimum_ev_cents

Example:
    >>> # 55% win probability, buying at 45 cents, 7% fee, min 1 cent EV
    >>> ExpectedValueGate.is_profitable_after_fees(0.55, 45, 100, 0.07, 1.0)
    True
    
    >>> # 50/50 at 50 cents with 7% fee = negative EV
    >>> ExpectedValueGate.is_profitable_after_fees(0.5, 50, 100, 0.07, 0.0)
    False
"""

from typing import Optional


class ExpectedValueGate:
    """
    Expected Value (EV) payout gating engine for prediction markets.
    
    This class provides a static method to evaluate whether a trade has
    positive expected value after accounting for exchange fees. It implements
    the core risk management logic that prevents deployment of negative-EV
    trades regardless of signal strength.
    
    The fee model assumes:
    - Fees are a percentage of the payout (not the stake)
    - Fees are capped at the maximum possible fee (exchange_fee_pct * payout)
    - Effective fee is the minimum of (gross_profit * fee_pct) and max_fee
    
    Attributes:
        None (stateless utility class)
    """
    
    @staticmethod
    def is_profitable_after_fees(
        predicted_win_prob: float,
        current_contract_price_cents: int,
        payout_cents: int = 100,
        exchange_fee_pct: float = 0.07,
        minimum_ev_cents: float = 1.0
    ) -> bool:
        """
        Determine if a trade has positive EV after fees.
        
        Args:
            predicted_win_prob: Model's predicted probability of winning (0.0-1.0).
            current_contract_price_cents: Current contract price in cents (0-100).
            payout_cents: Payout on win in cents (default 100 = $1.00).
            exchange_fee_pct: Exchange fee as decimal (default 0.07 = 7%).
            minimum_ev_cents: Minimum acceptable EV in cents (default 1.0).
            
        Returns:
            True if expected value >= minimum_ev_cents, False otherwise.
            
        Raises:
            ValueError: If predicted_win_prob not in [0, 1] or price not in [0, payout].
            
        Examples:
            >>> # Strong edge: 90% win prob at 10 cents, min EV 10 cents
            >>> ExpectedValueGate.is_profitable_after_fees(0.9, 10, 100, 0.07, 10.0)
            True
            
            >>> # Zero probability never profitable
            >>> ExpectedValueGate.is_profitable_after_fees(0.0, 50, 100, 0.07, 0.0)
            False
            
            >>> # 100% probability always profitable (if price < payout)
            >>> ExpectedValueGate.is_profitable_after_fees(1.0, 10, 100, 0.07, 0.0)
            True
            
            >>> # High fees kill marginal edges
            >>> ExpectedValueGate.is_profitable_after_fees(0.55, 45, 100, 0.50, 0.0)
            False
            
            >>> # Minimum EV threshold enforcement
            >>> ExpectedValueGate.is_profitable_after_fees(0.6, 40, 100, 0.07, 50.0)
            False
            
            >>> # Negative EV barrier allows suboptimal edges
            >>> ExpectedValueGate.is_profitable_after_fees(0.5, 50, 100, 0.07, -2.0)
            True
        """
        # Input validation
        if not 0.0 <= predicted_win_prob <= 1.0:
            raise ValueError(f"predicted_win_prob must be in [0, 1], got {predicted_win_prob}")
        if not 0 <= current_contract_price_cents <= payout_cents:
            raise ValueError(
                f"current_contract_price_cents must be in [0, {payout_cents}], "
                f"got {current_contract_price_cents}"
            )
        if exchange_fee_pct < 0:
            raise ValueError(f"exchange_fee_pct must be non-negative, got {exchange_fee_pct}")
        
        cost = current_contract_price_cents
        gross_profit = payout_cents - cost
        max_fee = exchange_fee_pct * payout_cents
        effective_fee = min(gross_profit * exchange_fee_pct, max_fee)
        net_profit_if_win = gross_profit - effective_fee
        loss_if_lose = -cost
        expected_value = (predicted_win_prob * net_profit_if_win) + ((1.0 - predicted_win_prob) * loss_if_lose)
        return expected_value >= minimum_ev_cents