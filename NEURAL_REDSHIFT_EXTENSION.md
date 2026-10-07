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

1. Sample random functions from matched network architectures.
2. Estimate their radial power spectra.
3. Compare finite networks with NNGP spectra.
4. Insert the measured spectra into this exact benchmark.
5. Test architecture effects under matched parameter counts.
