# Critical review and research plan

**Review date:** 5 October 2026  
**Status:** robustness tests complete

## Decision

Do not develop one universal ownership percentage.

Develop a control profile with separate measures for:

- prompt control;
- model and seed dependence;
- semantic compliance;
- causal image change;
- training-data counterfactual influence;
- human control judgments.

Each measure answers a different question.

## Important correction

The first experiment encoded unused routing bits as zero. Therefore, the seed code depended on the prompt mask.

This dependence gave some prompt information to the seed player. The corrected generator samples all routing bits independently.

The corrected two-attribute Shapley result is:

| Player | Corrected share |
|---|---:|
| Prompt | 40.8% |
| Model prior | 30.6% |
| Seed and routing | 28.6% |

The previous result was 34.6%, 33.1%, and 32.3%. Do not use the previous result.

## Main conceptual problems

### The result is global

Mutual information measures a population distribution. It does not measure one image.

The experiment answers this question:

> Across the defined prompt and seed population, how much information does each source provide about the output code?

It does not answer this question:

> Who created this one image?

A local score needs pointwise information or local Shapley values. These scores can be negative and unstable.

### The model player is not the trained model

Fixed weights are constants. Their mutual information is zero.

The model player is a sampled scene proposal. This variable represents prior behaviour, not stored weights or training data.

Training-data influence needs a separate counterfactual study. The correct test removes data and retrains or ablates the model.

### The gate is not production cross-attention

The gate is an attention-style proxy. It does not implement learned queries, keys, values, heads, layers, or diffusion steps.

Therefore, the experiment tests attribution logic. It does not validate diffusion attention.

### Shapley values include policy choices

Shapley values need a player set, baseline, utility, and coalition definition.

A prompt-only coalition cannot generate an image without a model. A real study must define a replacement model or baseline prompt.

### Partial information decomposition is not unique

The stress test uses factor-wise Williams–Beer \(I_{min}\). This measure has known copy problems.

Use it as a diagnostic. Do not treat it as a unique decomposition.

## New robustness results

### Output representation changes the result

At two prompt attributes:

| Output representation | Prompt | Model | Seed |
|---|---:|---:|---:|
| Semantics and background | 40.8% | 30.6% | 28.6% |
| Semantic attributes only | 53.9% | 40.3% | 5.8% |
| Background only | 0.0% | 0.0% | 100.0% |

The ratio is not a property of the image alone. It is a property of the selected representation.

### The denominator changes the public claim

The same two-attribute condition gives:

| Claim | Prompt share |
|---|---:|
| Prompt share of all three players | 40.8% |
| Prompt share after seed removal | 57.2% |
| Prompt versus grouped system | 40.6% |

The phrase “prompt share” is incomplete without the denominator.

### Prior entropy changes the allocation

| Model prior | Prior entropy | Prompt | Model | Seed |
|---|---:|---:|---:|---:|
| Concentrated | 1.101 bits | 56.3% | 8.6% | 35.1% |
| Learned correlated | 5.304 bits | 40.8% | 30.6% | 28.6% |
| Uniform | 6.340 bits | 39.4% | 32.4% | 28.2% |

A low-entropy model proposal gives less model information. The prompt then receives a larger share.

### Prompt population has a measurable effect

Uniform prompt values gave a 40.8% prompt share. Prior-aligned prompt values gave 40.2%.

The effect is small in this generator. It can be larger with stronger prompt and prior dependence.

### Finite samples create bias

| Samples | Mean absolute error across players |
|---:|---:|
| 128 | 2.60 percentage points |
| 512 | 2.50 percentage points |
| 2,048 | 2.21 percentage points |
| 8,192 | 1.26 percentage points |
| 32,768 | 0.39 percentage points |
| 131,072 | 0.07 percentage points |

Small repeated samples can give narrow intervals around a biased value. Bootstrap intervals do not remove estimator bias.

### Attention weights are not identifiable credit

Consider one attention output:

\[
o=a v.
\]

These pairs all give \(o=0.8\):

| Attention weight \(a\) | Value magnitude \(v\) | Output \(o\) |
|---:|---:|---:|
| 0.8 | 1.0 | 0.8 |
| 0.4 | 2.0 | 0.8 |
| 0.2 | 4.0 | 0.8 |

The output stays constant while the attention weight changes by four times. Attention weight alone cannot measure contribution.

## Research evidence added

The 2026 Nature Communications study uses leave-one-unit counterfactuals. It reports attribution decay as training sets increase.

The study also reports false attributions from image similarity. This supports a strict separation between resemblance and causal training-data influence.

The 2026 ACL Findings paper records Stable Diffusion cross-attention. It also uses fixed-seed token removal for causal validation.

This method supports the proposed next step. It does not support raw attention as an ownership ratio.

## Limits of the new literature

### Dai and Gifford, 2026

The paper gives a strong causal definition. It also has important limits.

- It studies leave-one-unit removal. It does not reject group or concept attribution.
- It uses a special ablatable ensemble. Standard production models can behave differently.
- Counterfactual radius depends on the image-distance measure.
- The largest training sets are much smaller than major production data sets.
- Claims about billion-image systems require extrapolation.

Use this paper to design training-data tests. Do not use it to declare all attribution impossible.

### Maliha and Hougen, 2026

The paper improves attention analysis with interventions. It also leaves open problems.

- Word deletion can change grammar and token positions.
- A fixed seed controls noise but does not remove all nonlinear interactions.
- Aggregation can hide differences between heads, layers, and steps.
- Stable Diffusion does not represent all current diffusion-transformer designs.

Use token replacement and activation patching with word deletion.

### AME and MMShapley, 2026

AME is a recent preprint, not an established standard. Its result depends on default coalition substitutes and CLIP-based utility.

The human study is small and has no objective ownership target. Treat AME as a proposal, not validation.

## Development plan

### Stage 1: finish the simulator

Status: complete.

- Keep prompt, model, and seed independent before generation.
- Report semantic and full-output representations.
- Report all denominator definitions.
- Report exact results and finite-sample estimates.
- Keep the attention reparameterisation control.

### Stage 2: run an open diffusion model

Use Stable Diffusion 1.5 first. Existing DAAM tools and published baselines support this model.

Use four removable prompt clauses and 32 fixed seeds. Generate all 16 prompt coalitions.

Test three guidance values and two samplers. Keep all other settings fixed.

Measure:

- CLIPScore for text and image alignment;
- TIFA or GenEval for semantic compliance;
- LPIPS for perceptual change;
- segmentation overlap for spatial grounding;
- DAAM maps for token location;
- fixed-seed token removal for causal effect.

Compute clause Shapley values for each metric. Do not create a prompt-only model coalition.

### Stage 3: test attention faithfulness

Record each layer, head, and diffusion step.

Compare attention with token removal, activation patching, and value-aware attention output.

Report calibration error, rank correlation, and failure examples. Do not report attention mass as credit.

### Stage 4: test training-data influence

Train a small ablatable diffusion ensemble. Use several training-set sizes.

Remove one image, creator, and semantic group. Keep the prompt and noise fixed.

Measure the counterfactual radius. Compare it with similarity and influence-function methods.

This stage tests training data. It must remain separate from prompt control.

### Stage 5: add human control judgments

Use blinded pairs and a registered protocol. Ask about control over specified expressive elements.

Do not ask participants to assign legal ownership. Compare human ratings with each technical measure.

## Success criteria

Continue development only if the study meets these conditions:

- The result is stable across seeds and samplers.
- The result is stable across at least two semantic metrics.
- Attention passes matched intervention tests.
- Each reported percentage names its denominator.
- The study separates prompt control from training-data influence.
- The report does not state a legal ownership threshold.

## Primary new sources

- Dai and Gifford, “Outputs of generative diffusion models are often unattributable,” *Nature Communications*, 18 August 2026: <https://www.nature.com/articles/s41467-026-75667-5>
- Maliha and Hougen, “Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions,” *Findings of ACL*, July 2026: <https://aclanthology.org/2026.findings-acl.1265/>
- Tang et al., “What the DAAM: Interpreting Stable Diffusion Using Cross Attention,” ACL 2023: <https://aclanthology.org/2023.acl-long.310/>
- U.S. Copyright Office, *Copyright and Artificial Intelligence, Part 2*, 29 January 2025: <https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf>
- Shi et al., “AME: A Multi-Type Contributor Attribution Framework in Generative AI Markets,” preprint, 15 June 2026: <https://arxiv.org/abs/2606.16075>

## Reproducibility

Run:

```bash
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiment.py
/Users/evam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 robustness.py
```

Main results SHA-256: `2a01dd8f740ff1a0dac3ce7db4455955b69c08bf13cba90e4596db0f45f1ddfe`  
Robustness results SHA-256: `ec03531ec5b8c6682e0d8571a82c7bd39c3cd84c1aec6ff9d9a47d489fe13266`
