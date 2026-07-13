from edge_mining_framework.gate import ExpectedValueGate


class TestExpectedValueGate:
    def test_breakeven_is_negative_with_fees(self):
        # 50/50 at 50 cents with 7% fee = negative EV
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=0.5,
            current_contract_price_cents=50,
            exchange_fee_pct=0.07,
            minimum_ev_cents=0.0,
        )
        assert result is False

    def test_strong_edge_is_profitable(self):
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=0.9,
            current_contract_price_cents=10,
            exchange_fee_pct=0.07,
            minimum_ev_cents=10.0,
        )
        assert result is True

    def test_zero_probability_never_profitable(self):
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=0.0,
            current_contract_price_cents=50,
            exchange_fee_pct=0.07,
            minimum_ev_cents=0.0,
        )
        assert result is False

    def test_100_pct_probability_always_profitable(self):
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=1.0,
            current_contract_price_cents=10,
            exchange_fee_pct=0.07,
            minimum_ev_cents=0.0,
        )
        assert result is True

    def test_high_fee_kills_marginal_edge(self):
        # Slight edge but 50% fees destroy it
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=0.55,
            current_contract_price_cents=45,
            exchange_fee_pct=0.50,
            minimum_ev_cents=0.0,
        )
        assert result is False

    def test_minimum_ev_threshold_enforced(self):
        # Profitable but below minimum EV threshold
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=0.6,
            current_contract_price_cents=40,
            exchange_fee_pct=0.07,
            minimum_ev_cents=50.0,
        )
        assert result is False

    def test_negative_ev_barrier_allows_suboptimal_edges(self):
        # EV is negative (-2.0), but minimum_ev_cents is -5.0
        # Expected value formula for this trade:
        # Cost: 40, Payout: 100, Gross Profit: 60, Fee: min(60*0.07=4.2, 7.0)=4.2
        # Net Profit if win: 55.8
        # EV = 0.45 * 55.8 + 0.55 * (-40) = 25.11 - 22.0 = 3.11 (wait, let's pick values for a negative EV)

        # Let's say cost = 50, win_prob = 0.5.
        # Net profit if win = 50 - 3.5 = 46.5
        # Loss if lose = -50
        # EV = 0.5 * 46.5 - 0.5 * 50 = 23.25 - 25 = -1.75
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=0.5,
            current_contract_price_cents=50,
            exchange_fee_pct=0.07,
            minimum_ev_cents=-2.0,
        )
        assert result is True

    def test_negative_ev_barrier_still_blocks_worse_edges(self):
        # Same -1.75 EV, but barrier is -1.0
        result = ExpectedValueGate.is_profitable_after_fees(
            predicted_win_prob=0.5,
            current_contract_price_cents=50,
            exchange_fee_pct=0.07,
            minimum_ev_cents=-1.0,
        )
        assert result is False
