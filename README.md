# V8 ⚡

QTS V8 — Quantitative Trading System

## Architecture

| Layer | Lang | Description |
|-------|------|-------------|
| **core** | Rust | Feature engine, MCTS core, OKX router, order state |
| **alpha** | Python | Cross-market, crypto, equity, forex, poly alpha strategies |
| **models** | Python/PyTorch | AlphaCast predictor, ResNet encoder, MCTS exploration |
| **execution** | Rust | Order routing, settlement channels |
| **ml** | Python | Meta-labeling pipeline |
| **schemas** | Proto | Alpha signal, market data, OKX order protos |

## Quick Start

\\\ash
# Rust core
cd core && cargo build --release

# Python ML
pip install -e .

# Run test suite
cargo test
pytest tests/
\\\

> Private repository. See /doc for full specifications.