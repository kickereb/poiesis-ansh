# Poiesis Ansh

*Poiesis* means creation. *Ansh* means share or contribution.

The project measures each factor's declared contribution to a generated image.

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

Run the exact Gaussian spectral benchmark:

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 spectral_theory.py
```

Then read [`NEURAL_REDSHIFT_EXTENSION.md`](NEURAL_REDSHIFT_EXTENSION.md).

Run the random-function architecture pilot:

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 architecture_prior.py
```

Run its tests:

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests
```

The real Stable Diffusion experiment is in [`real_model/`](real_model/README.md).
Its methods and sources are reviewed in [`REAL_MODEL_LITERATURE_REVIEW.md`](REAL_MODEL_LITERATURE_REVIEW.md).
Its completed pilot report is [`REAL_MODEL_REPORT.md`](REAL_MODEL_REPORT.md).

The transparent experiment requires only NumPy and Pillow.
It tests estimators in a learned-prior generator.
It does not measure legal ownership or directly attribute fixed neural-network weights.
