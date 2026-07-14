"""Cross-package checks for the installed backtest-harness boundary."""

import pytest


@pytest.mark.integration
def test_backtest_harness_public_adapter() -> None:
    """Use only the installed package surface when the sibling package is present."""
    backtest_harness = pytest.importorskip("backtest_harness")
    returns, provenance = backtest_harness.load_kalshi_sample_trades()
    stats = backtest_harness.MonteCarloSimulator.simulate_equity_paths(
        trade_returns_pct=returns,
        starting_equity=1_000.0,
        num_simulations=100,
        trades_per_sim=returns.size,
        seed=20_260_713,
    )

    assert provenance["source"] == "synthetic"
    assert stats["used_numba"] is True
