import pytest
import urllib.request
import urllib.error
import json
from edge_mining_framework.evaluator import FeatureEvaluator

def test_live_data_pipeline():
    """
    REAL INTEGRATION TEST.
    Instead of mocked dicts, fetches LIVE Bitcoin volume/price data from Binance's public API,
    translates it into a feature map, and evaluates a compound rule against the live market.
    """
    try:
        req = urllib.request.Request('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req) as response:
            live_data = json.loads(response.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        pytest.skip(f"Binance API is restricted or network error: {e}")

    
    # 2. Extract Features
    features = {
        "btc_price": float(live_data["lastPrice"]),
        "price_change_pct": float(live_data["priceChangePercent"]),
        "volume": float(live_data["volume"])
    }
    
    # 3. Define a real ruleset (e.g. BTC > 10,000)
    rules = [
        {"feature": "btc_price", "operator": ">", "threshold": 10000.0}
    ]
    
    # 4. Evaluate
    result = FeatureEvaluator.evaluate_compound(features, rules)
    assert result is True  # BTC should definitely be > 10k right now
