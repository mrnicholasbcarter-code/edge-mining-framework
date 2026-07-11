class ExpectedValueGate:
    """
    Financial entry gate calculating real Expected Value (EV).
    Prevents deploying capital when exchange friction exceeds alpha.
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
        Calculates EV specifically mapped to continuous prediction markets 
        (e.g., Kalshi's variable 7% fee ceiling model).
        """
        cost = current_contract_price_cents
        gross_profit = payout_cents - cost
        
        # Fee model: standard is % of gross profit (bounded)
        max_fee = exchange_fee_pct * payout_cents
        effective_fee = min(gross_profit * exchange_fee_pct, max_fee)
        
        net_profit_if_win = gross_profit - effective_fee
        loss_if_lose = -cost
        
        expected_value = (predicted_win_prob * net_profit_if_win) + ((1.0 - predicted_win_prob) * loss_if_lose)
        
        return expected_value >= minimum_ev_cents
