"""Use the installed backtest-harness package through its public API.

This example intentionally has no sibling-path or ``sys.path`` fallback. Install
the dependency first, then run the example from this repository.
"""

from __future__ import annotations

from backtest_harness import MonteCarloSimulator, load_kalshi_sample_trades, tearsheet


def main() -> None:
    """Print a small deterministic report from the installed backtest package."""
    returns, provenance = load_kalshi_sample_trades()
    stats = MonteCarloSimulator.simulate_equity_paths(
        trade_returns_pct=returns,
        starting_equity=1_000.0,
        num_simulations=1_000,
        trades_per_sim=returns.size,
        seed=20_260_713,
    )
    summary = tearsheet(returns)
    assert summary is not None

    print("Installed backtest-harness adapter")
    print(f"  Fixture: {provenance['name']}")
    print(f"  P50 final equity: ${stats['p50_equity']:.2f}")
    print(f"  Risk of ruin: {stats['prob_ruin']:.2%}")
    print(f"  Max drawdown: {summary['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
