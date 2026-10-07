# Poiesis Ansh: Neural Redshift extension

## Phase 1 status

The exact Gaussian spectral benchmark is implemented.

The benchmark separates prompt, model-prior, and seed contributions by frequency band.
It uses exact mutual information for jointly Gaussian sources.
It then applies exact three-player Shapley allocation.

## Model

For frequency band \(k\):

\[
Y_k=P_k+M_k+Z_k+\epsilon_k.
\]

The source vector is jointly Gaussian.
The observation noise is independent Gaussian noise.

For coalition \(S\):

\[
v_k(S)=I(X_{S,k};Y_k).
\]

The total game is additive across independent frequency bands:

\[
v(S)=\sum_k v_k(S).
\]

## Included conditions

- Equal white spectra.
- Low-frequency model prior.
- Low-frequency prompt.
- Frequency-separated prompt and model.
- Seed-dominant output.
- Correlated prompt and model signals.

All conditions use equal total prompt and model variance unless stated otherwise.

## Initial results

| Condition | Prompt | Model prior | Seed |
|---|---:|---:|---:|
| Equal white spectra | 33.3% | 33.3% | 33.3% |
| Low-frequency model prior | 43.9% | 12.3% | 43.9% |
| Low-frequency prompt | 12.3% | 43.9% | 43.9% |
| Frequency-separated sources | 38.3% | 14.7% | 47.0% |
| Seed-dominant output | 25.6% | 12.7% | 61.8% |
| Correlated prompt and model | 32.0% | 18.8% | 49.1% |

Equal total variance did not give equal credit after spectral concentration.
Concentrated power entered fewer bands and met stronger information saturation.
This result depends on the declared noise model and utility.
It is not a general result for neural networks.

## Interpretation

The profile shows where each source contributes information.
It does not show legal ownership.

The architecture will remain an experimental condition in the next phase.
Random functions from that architecture will become the model-prior population.

## Run

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 spectral_theory.py
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests
```

## Next phase

Phase 2 completed the first, second, fourth, and fifth items.

## Phase 2: random-function architecture priors

The pilot sampled 128 untrained functions from each of eight architectures.
Each function produced a 32 by 32 scalar image.
Parameter counts remained within 8.6% of the 4,096-parameter target.

The suite tested:

- ReLU, tanh, and Gaussian coordinate MLPs.
- A ReLU MLP with non-affine layer normalisation.
- A sinusoidal representation network.
- A ReLU MLP with fixed Fourier features.
- Plain and residual convolutional networks.

Every sampled image was centred and scaled to unit variance.
This control removes amplitude as an explanation for spectral differences.

### Results

| Architecture | Low-frequency energy | Spectral centre | Model-prior Shapley share |
|---|---:|---:|---:|
| ReLU MLP | 90.7% | 0.126 | 18.9% |
| Tanh MLP | 91.2% | 0.123 | 18.5% |
| Gaussian MLP | 95.0% | 0.100 | 15.6% |
| ReLU and layer normalisation | 92.8% | 0.113 | 17.4% |
| Sinusoidal network | 83.0% | 0.199 | 23.0% |
| Fourier-feature ReLU MLP | 56.1% | 0.325 | 28.0% |
| Convolutional ReLU | 90.7% | 0.128 | 19.1% |
| Residual convolutional ReLU | 90.2% | 0.130 | 19.3% |

The common coordinate and convolutional networks strongly favoured low frequencies.
Fourier features produced the broadest spectrum in this suite.

The measured model-prior share ranged from 15.6% to 28.0%.
Prompt and seed spectra remained uniform for this comparison.
All three sources had equal total variance.

More distributed spectra entered more independent bands in the Gaussian game.
They therefore received more aggregate information credit under this utility.

### Limits

This pilot is not a direct replication of the paper's Boolean-function experiments.
It is a two-dimensional spectral extension of the paper's random-function principle.

The result depends on:

- Coordinate encoding.
- Initialisation distribution.
- Activation parameters.
- Image resolution.
- Spectrum estimator.
- Output normalisation.
- Gaussian attribution utility.

The Shapley result measures model-prior information.
It does not measure ownership.

Primary paper: [Teney et al., Neural Redshift, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Teney_Neural_Redshift_Random_Networks_are_not_Random_Functions_CVPR_2024_paper.html).

## Next phase

1. Derive NNGP kernels for the coordinate MLP conditions.
2. Compare finite-width spectra with infinite-width samples.
3. Add width and depth convergence sweeps.
4. Separate prior spectra from NTK training dynamics.
