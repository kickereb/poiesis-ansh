# Real-model influence experiment

This directory contains the Stable Diffusion extension of Poiesis Ansh.

The earlier percentages came from two transparent systems:

1. An exactly enumerable categorical image generator.
2. Random-network Gaussian spectral benchmarks.

They did not come from Stable Diffusion.

Those systems tested estimator behaviour under known ground truth.
This directory tests a real open-weight model.

## Model

- Model: `CompVis/stable-diffusion-v1-4`
- Revision: `133a221b8aa7292a167afc5127cb63fb5005638b`
- Licence: CreativeML OpenRAIL-M
- Scheduler: DDIM
- Device: Apple MPS
- Primary precision: FP32

The weights stay in `real_model/cache/` and are not tracked by Git.
The model manifest records exact file hashes.

## Environment

Use only the isolated interpreter:

```bash
real_model/.venv/bin/python
```

Rebuild it with:

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv real_model/.venv
real_model/.venv/bin/python -m pip install -r real_model/requirements.txt
```

Download the pinned public model snapshot:

```bash
PYTHONPATH=real_model real_model/.venv/bin/python real_model/download_model.py
```

## Runs

Baseline:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 real_model/.venv/bin/python real_model/run_baseline.py
```

Activation patching:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 real_model/.venv/bin/python real_model/run_activation_patching.py
```

Weight directions and low-rank doses:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 real_model/.venv/bin/python real_model/run_weight_directions.py
```

Selected weight gradients and a gradient-aligned rank-4 dose sweep:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 real_model/.venv/bin/python real_model/run_gradient_svd_lora.py
```

Final float-image gradients for selected weights:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 real_model/.venv/bin/python real_model/run_output_weight_gradients.py
```

Train and test one controlled rank-4 LoRA adapter:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 real_model/.venv/bin/python real_model/run_trained_lora.py
```

Test fixed interventions on one fitting seed and three held-out seeds:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 real_model/.venv/bin/python real_model/run_seed_robustness.py
```

Build the combined result summary:

```bash
PYTHONPATH=real_model real_model/.venv/bin/python real_model/summarize_results.py
```

Tests:

```bash
PYTHONPATH=real_model real_model/.venv/bin/python -m unittest discover -s real_model/tests -v
```

## Interpretation

- Gradients measure local sensitivity.
- Low-rank doses measure one controlled weight-direction effect.
- Activation patches measure the effect of replacing one complete internal state under one prompt contrast.
- Attention maps show spatial association unless interventions validate them.
- None of these methods measures legal ownership.
