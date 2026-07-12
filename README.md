<div align="center">
  <h1>Edge Mining Framework</h1>
  <p><strong>Agnostic Signal Feature Evaluator and EV Payout Gating Engine</strong></p>
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status" />
  <img src="https://img.shields.io/badge/signals-fail--fast-orange" alt="Signals" />
</div>

## Architecture

This repository evaluates live orderbook/alpha features against strict compound thresholds and mathematically blocks deployment unless the Expected Value (EV) exceeds friction parameters.

```mermaid
sequenceDiagram
    participant Pipeline
    participant FeatureEvaluator
    participant ExpectedValueGate
    
    Pipeline->>FeatureEvaluator: evaluate_compound(features: DICT, rules: YAML)
    alt Signal Miss
        FeatureEvaluator-->>Pipeline: False (Abort)
    else Signal Match
        FeatureEvaluator->>ExpectedValueGate: is_profitable_after_fees(prob: 0.55, fees: 7%)
        alt Negative EV
            ExpectedValueGate-->>Pipeline: False (Friction too high)
        else Positive Alpha
            ExpectedValueGate-->>Pipeline: True (Execute Order)
        end
    end
```

## Modular Operators

The `FeatureEvaluator` utilizes a dictionary routing technique against Python's base `operator` module, avoiding slow and unsafe `eval()` operations.

Supported O(1) mathematical operators:
`==`, `!=`, `<`, `>`, `<=`, `>=`, `in_range`, `in_set`
