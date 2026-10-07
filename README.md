# Image attribution experiment

This directory contains a complete, deterministic toy experiment comparing mutual-information decomposition, exact Shapley allocation, and a cross-attention-style mechanistic probe for prompt-conditioned image generation.

Run:

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiment.py
```

Then read [`REPORT.md`](REPORT.md). Generated files are written to `artifacts/`.

Run the robustness study:

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 robustness.py
```

Then read [`CRITIQUE_AND_RD.md`](CRITIQUE_AND_RD.md).

The experiment requires only NumPy and Pillow. It measures estimator behaviour in a transparent learned-prior generator; it does not claim a legal ownership percentage or a direct attribution of fixed neural-network weights.
