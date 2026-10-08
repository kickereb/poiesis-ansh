# Real-model influence literature review

**Review date:** 9 October 2026

**Scope:** Stable Diffusion parameter sensitivity, low-rank weight changes, activation patching, attention, and causal limits

## Finding

The literature supports four complementary methods.

| Method | Supported interpretation |
|---|---|
| Parameter gradient | Local sensitivity of a declared objective |
| LoRA-form low-rank change | Effect of one low-rank weight direction |
| Trained LoRA | Effect of one learned low-rank adapter |
| Activation patch | Interventional effect under a declared contrast |

These methods do not measure legal ownership.

## Parameter gradients

[Cones](https://proceedings.mlr.press/v202/liu23j/liu23j.pdf) identifies concept-related Stable Diffusion parameters with gradient statistics.
It then tests selected attention parameters through intervention.

[SalUn](https://arxiv.org/abs/2310.12508) uses loss gradients to select weights for model unlearning.

[D-TRAK](https://arxiv.org/abs/2311.00500) uses projected parameter gradients for diffusion training-data attribution.
Its results also show strong dependence on the selected gradient objective.

[Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) gives axioms for path-based attribution.
Its result still depends on the baseline and path.
It applies to input features, not model parameters.
This project uses it only as an analogy.

This project proposes the following extension for weight group \(g\) and region \(R\):

\[
S_{g,R}=\left\|\nabla_{\theta_g}J_R(X)\right\|.
\]

This quantity measures local sensitivity.
It depends on the parameter coordinate system.

The pilot reports one denoiser-objective gradient at 512×512 pixels.
It also reports a final float-RGB scalar gradient at 256×256 pixels.
Regional finite differences test selected rank-4 directions.
These finite differences do not estimate the full regional Jacobian norm.

## Low-rank weight changes

[Concept Sliders](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf) learns continuous low-rank semantic directions.
It also measures unwanted changes to other attributes.

[Interpreting the Weight Space of Customized Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f85364507054c257959c2011c28bfc0d-Abstract-Conference.html) finds structured directions across customised diffusion models.

[SliderSpace](https://openaccess.thecvf.com/content/ICCV2025/html/Gandikota_SliderSpace_Decomposing_the_Visual_Capabilities_of_Diffusion_Models_ICCV_2025_paper.html) discovers interpretable low-rank directions without one manually named attribute per direction.

The clean intervention is:

\[
\theta(\alpha)=\theta_0+\alpha\Delta\theta.
\]

The experiment holds the prompt, seed, sampler, and scheduler fixed.
It tests negative, zero, and positive doses.
The gradient-SVD rank-4 change has LoRA form, but training did not produce it.
A separate experiment trains LoRA factors on one declared denoiser objective.
The cited work trains broader semantic adapter directions.

Required controls include:

- Rank-matched random directions.
- Norm-matched random directions.
- Zero-strength equality.
- Positive and negative doses.
- Target and non-target region changes.

## Activation patching

[Localizing and Editing Knowledge in Text-to-Image Generative Models](https://proceedings.iclr.cc/paper_files/paper/2024/file/4bfcebedf7a2967c410b64670f27f904-Paper-Conference.pdf) adapts causal tracing to Stable Diffusion.
It reports distributed U-Net effects and more local text-encoder effects.

[LocoGen](https://proceedings.mlr.press/v235/basu24b.html) finds that localisation does not transfer uniformly across model families.
It supports direct interventions over general claims from one architecture.

[Precise Parameter Localization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/add6cccf7ddc718eb6a7991cb531812d-Abstract-Conference.html) uses attention activation patching to locate rendered-text control.

[Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions](https://aclanthology.org/2026.findings-acl.1265/) records token-level cross-attention signals.
It tests them with fixed-seed, one-token prompt interventions.

A restoring patch tests sufficiency under the selected corruption.
A disruptive patch tests necessity under the selected intervention.
Neither test proves exclusive representation.

[An Interpretability Illusion for Subspace Activation Patching](https://openreview.net/forum?id=Ebt7JgMHv1) shows that a patch can use inactive routes.
This result limits mechanistic claims from patching alone.

Required controls include:

- Source-to-source identity patches.
- Target-to-target identity patches.
- Unrelated donors.
- Wrong regions.
- Channel permutations.
- Both patching directions.
- Partial patch strengths.

## Attention maps

[DAAM](https://aclanthology.org/2023.acl-long.310/) aggregates cross-attention into token-region maps.
It reports noun segmentation performance between 58.8% and 64.8% mean intersection over union.

[Prompt-to-Prompt](https://openreview.net/pdf/a6e78444f28f4790c2b8eb24364ced3ce736feb0.pdf) shows that cross-attention injection can preserve or change layout.

[Attention Is Not Explanation](https://aclanthology.org/N19-1357/) shows that attention can disagree with gradients.
Alternative attention maps can also preserve model output.
That study uses recurrent language models.
It gives a general warning, not direct Stable Diffusion evidence.

This project treats attention as descriptive evidence.
Matched interventions must support causal statements.

## Sparse features

[Emergence and Evolution of Interpretable Concepts in Diffusion Models](https://arxiv.org/abs/2504.15473) finds spatial semantic features in Stable Diffusion activations.
Its interventions associate early stages with composition, middle stages with style, and late stages with texture.

[DIFFLENS](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_Dissecting_and_Mitigating_Diffusion_Bias_via_Mechanistic_Interpretability_CVPR_2025_paper.pdf) uses sparse features and interventions to study diffusion bias.

[SAeUron](https://arxiv.org/abs/2501.18052) suppresses objects and styles by blocking selected sparse features.

[Look But Do Not Touch](https://arxiv.org/abs/2606.31699) reports that sparse-feature steering can create abnormal activations and image defects.

Sparse autoencoders belong after basic patching validation.
Genuine activation patches provide an important control.

## Training-data influence

Model-component influence is not training-image influence.

[Data Attribution by Unlearning Synthesized Images](https://proceedings.neurips.cc/paper_files/paper/2024/hash/07fbde96bee50f4e09303fd4f877c2f3-Abstract-Conference.html) defines influence through exact removal and retraining.
Its practical method approximates this test through synthesized-image unlearning.

[Outputs of Generative Diffusion Models Are Often Unattributable](https://www.nature.com/articles/s41467-026-75667-5) reports declining individual-data attribution as training sets grow.
It also reports false attribution from visual similarity.

Training-data claims require removal, retraining, or a validated approximation.

## Philosophical limits

The methods separate:

- Statistical dependence.
- Local sensitivity.
- Counterfactual causation.
- Mechanistic constitution.
- Normative ownership.

A fixed model can have zero mutual information with output variation.
It can still remain causally and mechanically necessary.

Influence depends on the comparison baseline.
It also depends on whether the unit is a weight, head, block, checkpoint, or complete system.

Redundant pathways can hide causal importance under ablation.
Artificial patches can also use routes that normal generation does not use.

Mechanistic contribution does not establish intention, agency, authorship, desert, or ownership.
