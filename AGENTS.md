# AI Agent Constraints & Architectural Context (.claude.md)

**Target Ecosystem:** Alpha Discovery & Signal Evaluation
**Language:** Python 3.10+
**Primary Directive:** Modular, uncoupled feature gating.

## Core Directives
- **Feature Evaluator Isolation:** `FeatureEvaluator` is strictly agnostic to where data comes from. Do not import `KalshiClient` or any exchange SDKs into this repository. It evaluates raw dictionaries.
- **Fail-Fast Loops:** When evaluating compound rulesets, ensure internal algorithms break and return `False` immediately upon the first failed condition to preserve CPU cycles on downstream feature resolution.

## Edge Testing
When an AI agent modifies `ExpectedValueGate`, it must create an exact mathematical unit test in `tests/test_ev_gate.py` bounding the theoretical loss limit of the execution against variable exchange fees.
