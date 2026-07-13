import json
import os
import urllib.error
import urllib.request

import pytest

from edge_mining_framework.evaluator import FeatureEvaluator


def _evaluate_ticker_payload(live_data: dict[str, object]) -> bool:
    """Translate a ticker payload at the provider boundary and evaluate a rule."""
    features = {
        "btc_price": float(live_data["lastPrice"]),
        "price_change_pct": float(live_data["priceChangePercent"]),
        "volume": float(live_data["volume"]),
    }
    rules = [{"feature": "btc_price", "operator": ">", "threshold": 10000.0}]
    return FeatureEvaluator.evaluate_compound(features, rules)


def test_live_data_pipeline_fixture() -> None:
    """Exercise the provider-to-feature boundary without network dependence."""
    fixture = {
        "lastPrice": "65000.00",
        "priceChangePercent": "1.25",
        "volume": "12345.67",
    }

    assert _evaluate_ticker_payload(fixture) is True


@pytest.mark.skipif(
    os.getenv("EDGE_MINING_LIVE_INTEGRATION") != "1",
    reason="set EDGE_MINING_LIVE_INTEGRATION=1 to enable the optional network check",
)
def test_optional_live_data_pipeline() -> None:
    """Optionally validate the same boundary against Binance's public API."""
    try:
        req = urllib.request.Request(
            "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req) as response:
            live_data = json.loads(response.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        pytest.skip(f"Binance API is restricted or network error: {e}")

    assert _evaluate_ticker_payload(live_data) is True
