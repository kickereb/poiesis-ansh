#!/usr/bin/env python3
"""Exact toy experiment for prompt/model/noise attribution in image generation.

The experiment intentionally uses a transparent categorical generator rather than
claiming that a black-box image model exposes an ownership percentage. A learned
scene prior proposes four visual attributes. Prompt tokens probabilistically
override those proposals through an attention-style routing gate. A seed controls
both routing and background microtexture. Because all variables are discrete, the
mutual-information game and its Shapley values can be evaluated by exact
enumeration.

Dependencies: numpy and Pillow only.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SEED = 20260926
TRAINING_ROWS = 30_000
DIRICHLET_ALPHA = 0.5

ATTR_NAMES = ("shape", "colour", "position", "texture")
ATTR_VALUES = (
    ("circle", "square", "triangle"),
    ("red", "green", "blue"),
    ("left", "centre", "right"),
    ("solid", "stripes", "dots"),
)

# Prompt-token attention probabilities for the four attribute slots. These are
# deliberately imperfect: a specified token can fail to override the model prior.
BASE_GUIDANCE = np.array([0.96, 0.91, 0.86, 0.81], dtype=float)

PLAYER_NAMES = ("prompt", "model_prior", "seed_noise")
def all_attribute_tuples() -> np.ndarray:
    return np.array(list(itertools.product(range(3), repeat=4)), dtype=np.int16)


ALL_TUPLES = all_attribute_tuples()


def true_prior_probabilities() -> np.ndarray:
    """Return the correlated population distribution used to create training data."""
    probs = []
    for shape, colour, position, texture in ALL_TUPLES:
        p_shape = 1.0 / 3.0
        # The corpus favours red circles, green squares, and blue triangles.
        p_colour = 0.74 if colour == shape else 0.13
        # Shape also weakly predicts where an object appears.
        p_position = 0.58 if position == shape else 0.21
        # Colour predicts a preferred surface texture.
        p_texture = 0.68 if texture == colour else 0.16
        probs.append(p_shape * p_colour * p_position * p_texture)
    out = np.asarray(probs, dtype=float)
    return out / out.sum()


def learn_model_prior(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a smoothed joint categorical prior from a finite synthetic corpus."""
    true_probs = true_prior_probabilities()
    draws = rng.choice(len(ALL_TUPLES), size=TRAINING_ROWS, p=true_probs)
    counts = np.bincount(draws, minlength=len(ALL_TUPLES)).astype(float)
    learned = (counts + DIRICHLET_ALPHA) / (
        counts.sum() + DIRICHLET_ALPHA * len(counts)
    )
    return counts.astype(np.int64), learned


def encode_base(rows: np.ndarray, base: int) -> np.ndarray:
    """Encode rows of equally based digits as integers."""
    rows = np.asarray(rows, dtype=np.int64)
    powers = (base ** np.arange(rows.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum(rows * powers, axis=1, dtype=np.int64)


def encode_prompt(rows: np.ndarray) -> np.ndarray:
    # Prompt uses -1 for an omitted attribute. Shift to 0..3 before base-4 coding.
    return encode_base(np.asarray(rows, dtype=np.int64) + 1, 4)


def weighted_entropy(codes: np.ndarray, weights: np.ndarray) -> float:
    """Shannon entropy (bits) of integer codes under exact probability weights."""
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    sorted_weights = weights[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1])]
    probs = np.add.reduceat(sorted_weights, starts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def coalition_code(
    coalition: Tuple[int, ...], source_codes: Sequence[np.ndarray]
) -> np.ndarray:
    if not coalition:
        return np.zeros_like(source_codes[0], dtype=np.int64)
    code = np.zeros_like(source_codes[0], dtype=np.int64)
    multiplier = 1
    for player in coalition:
        code += source_codes[player] * multiplier
        multiplier *= int(source_codes[player].max()) + 1
    return code


def information_game(
    source_codes: Sequence[np.ndarray], x_code: np.ndarray, weights: np.ndarray
) -> Dict[Tuple[int, ...], float]:
    """Compute v(S)=I(S;X) for every source coalition."""
    h_x = weighted_entropy(x_code, weights)
    values: Dict[Tuple[int, ...], float] = {(): 0.0}
    for size in range(1, 4):
        for coalition in itertools.combinations(range(3), size):
            s_code = coalition_code(coalition, source_codes)
            h_s = weighted_entropy(s_code, weights)
            x_card = int(x_code.max()) + 1
            joint = s_code.astype(np.int64) * x_card + x_code
            h_sx = weighted_entropy(joint, weights)
            value = h_s + h_x - h_sx
            if value < -1e-8:
                raise AssertionError(f"negative mutual information: {value}")
            values[coalition] = max(0.0, value)
    return values


def shapley_values(game: Mapping[Tuple[int, ...], float]) -> np.ndarray:
    """Exact Shapley values for the three-player information game."""
    n = 3
    phis = np.zeros(n, dtype=float)
    all_players = set(range(n))
    for i in range(n):
        others = sorted(all_players - {i})
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                coefficient = (
                    math.factorial(size)
                    * math.factorial(n - size - 1)
                    / math.factorial(n)
                )
                with_i = tuple(sorted(subset + (i,)))
                phis[i] += coefficient * (game[with_i] - game[tuple(subset)])
    return phis


def prompt_states(specificity: int) -> List[np.ndarray]:
    states: List[np.ndarray] = []
    for mask in itertools.combinations(range(4), specificity):
        for values in itertools.product(range(3), repeat=specificity):
            p = np.full(4, -1, dtype=np.int16)
            for attr, value in zip(mask, values):
                p[attr] = value
            states.append(p)
    return states


def routing_states(guidance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Enumerate seed routing bits independently of the prompt."""
    routes: List[np.ndarray] = []
    probs: List[float] = []
    for bits in itertools.product((0, 1), repeat=4):
        route = np.asarray(bits, dtype=np.int16)
        probability = 1.0
        for attr, bit in enumerate(bits):
            probability *= guidance[attr] if bit else (1.0 - guidance[attr])
        routes.append(route)
        probs.append(probability)
    return np.asarray(routes, dtype=np.int16), np.asarray(probs, dtype=float)


def enumerate_experiment(
    specificity: int, model_probs: np.ndarray, guidance: np.ndarray
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate every non-zero (P,M,Z,X) state and its exact probability."""
    prompts = prompt_states(specificity)
    p_probability = 1.0 / len(prompts)
    model_codes = encode_base(ALL_TUPLES, 3)

    p_blocks: List[np.ndarray] = []
    m_blocks: List[np.ndarray] = []
    z_blocks: List[np.ndarray] = []
    x_blocks: List[np.ndarray] = []
    w_blocks: List[np.ndarray] = []

    for prompt in prompts:
        routes, route_probs = routing_states(guidance)
        # Cross each route with four visible seed/microtexture states.
        route_rows = np.repeat(routes, 4, axis=0)
        noise_ids = np.tile(np.arange(4, dtype=np.int16), len(routes))
        z_probs = np.repeat(route_probs, 4) / 4.0
        n_z = len(noise_ids)

        m_rows = np.repeat(ALL_TUPLES, n_z, axis=0)
        routed = np.tile(route_rows, (len(ALL_TUPLES), 1))
        noises = np.tile(noise_ids, len(ALL_TUPLES))
        y_rows = m_rows.copy()
        for attr in range(4):
            use_prompt = (prompt[attr] >= 0) & (routed[:, attr] == 1)
            y_rows[use_prompt, attr] = prompt[attr]

        block_size = len(m_rows)
        p_blocks.append(np.full(block_size, encode_prompt(prompt.reshape(1, -1))[0]))
        m_blocks.append(np.repeat(model_codes, n_z))
        route_codes = encode_base(routed, 2)
        z_blocks.append(route_codes * 4 + noises)
        x_blocks.append(encode_base(y_rows, 3) * 4 + noises)
        w_blocks.append(
            p_probability * np.repeat(model_probs, n_z) * np.tile(z_probs, len(model_probs))
        )

    source_codes = [
        np.concatenate(p_blocks),
        np.concatenate(m_blocks),
        np.concatenate(z_blocks),
    ]
    x_code = np.concatenate(x_blocks)
    weights = np.concatenate(w_blocks)
    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise AssertionError(f"enumerated probability sums to {weights.sum()}")
    return source_codes, x_code, weights, np.asarray(prompts, dtype=np.int16)


def decode_scene_code(x_code: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    noise = x_code % 4
    scene = x_code // 4
    attrs = np.zeros((len(scene), 4), dtype=np.int16)
    for attr in range(4):
        attrs[:, attr] = scene % 3
        scene //= 3
    return attrs, noise.astype(np.int16)


def render_scene(attrs: Sequence[int], noise_id: int, size: int = 64) -> Image.Image:
    """Render one categorical output into a small, inspectable RGB image."""
    shape, colour, position, texture = map(int, attrs)
    palettes = ((218, 63, 70), (53, 154, 99), (55, 112, 190))
    backgrounds = ((247, 247, 244), (241, 244, 247), (248, 244, 235), (239, 243, 239))
    image = Image.new("RGB", (size, size), backgrounds[noise_id])
    bg = ImageDraw.Draw(image)
    if noise_id == 1:
        for y in range(4, size, 8):
            bg.line((0, y, size, y), fill=(225, 230, 235), width=1)
    elif noise_id == 2:
        for y in range(5, size, 10):
            for x in range(5, size, 10):
                bg.point((x, y), fill=(229, 222, 207))
    elif noise_id == 3:
        for y in range(0, size, 8):
            for x in range(0, size, 8):
                if (x // 8 + y // 8) % 2 == 0:
                    bg.rectangle((x, y, x + 7, y + 7), fill=(231, 237, 231))

    x_centres = (18, 32, 46)
    cx = x_centres[position] + ((noise_id % 3) - 1)
    cy = 32 + ((noise_id // 2) - 1)
    radius = 13
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    if shape == 0:
        md.ellipse(bbox, fill=255)
    elif shape == 1:
        md.rounded_rectangle(bbox, radius=3, fill=255)
    else:
        md.polygon(((cx, cy - radius - 2), (cx - radius - 1, cy + radius), (cx + radius + 1, cy + radius)), fill=255)

    fill_layer = Image.new("RGB", (size, size), palettes[colour])
    image.paste(fill_layer, (0, 0), mask)

    if texture != 0:
        texture_layer = Image.new("RGB", (size, size), (0, 0, 0))
        td = ImageDraw.Draw(texture_layer)
        ink = (255, 255, 255)
        if texture == 1:
            for offset in range(-size, size * 2, 7):
                td.line((offset, 0, offset - size, size), fill=ink, width=2)
        else:
            for y in range(6, size, 8):
                for x in range(6, size, 8):
                    td.ellipse((x - 1, y - 1, x + 1, y + 1), fill=ink)
        # Texture is lightened before compositing to preserve the base colour.
        texture_layer = Image.blend(fill_layer, texture_layer, 0.28)
        image.paste(texture_layer, (0, 0), mask)

    outline = ImageDraw.Draw(image)
    if shape == 0:
        outline.ellipse(bbox, outline=(45, 49, 52), width=2)
    elif shape == 1:
        outline.rounded_rectangle(bbox, radius=3, outline=(45, 49, 52), width=2)
    else:
        outline.line(
            ((cx, cy - radius - 2), (cx - radius - 1, cy + radius), (cx + radius + 1, cy + radius), (cx, cy - radius - 2)),
            fill=(45, 49, 52),
            width=2,
        )
    return image


def renderer_feature_energy() -> np.ndarray:
    """Estimate each semantic slot's mean pixel effect by exhaustive interventions."""
    effects = np.zeros(4, dtype=float)
    counts = np.zeros(4, dtype=int)
    for attr in range(4):
        other_attrs = [j for j in range(4) if j != attr]
        for context_values in itertools.product(range(3), repeat=3):
            base = np.zeros(4, dtype=np.int16)
            for j, value in zip(other_attrs, context_values):
                base[j] = value
            rendered = []
            for value in range(3):
                candidate = base.copy()
                candidate[attr] = value
                rendered.append(np.asarray(render_scene(candidate, 0), dtype=float) / 255.0)
            for a, b in itertools.combinations(range(3), 2):
                effects[attr] += float(np.mean((rendered[a] - rendered[b]) ** 2))
                counts[attr] += 1
    effects /= counts
    return effects / effects.sum()


def expected_attention_share(prompts: np.ndarray, guidance: np.ndarray, energy: np.ndarray) -> float:
    shares = []
    for prompt in prompts:
        specified = prompt >= 0
        shares.append(float(np.sum(energy[specified] * guidance[specified])))
    return float(np.mean(shares))


def expected_causal_prompt_change(
    source_codes: Sequence[np.ndarray], x_code: np.ndarray, weights: np.ndarray, energy: np.ndarray
) -> float:
    """Expected semantic pixel-energy changed by ablating every prompt token."""
    model_code = source_codes[1].copy()
    model_attrs = np.zeros((len(model_code), 4), dtype=np.int16)
    for attr in range(4):
        model_attrs[:, attr] = model_code % 3
        model_code //= 3
    out_attrs, _ = decode_scene_code(x_code)
    changed = np.sum((out_attrs != model_attrs) * energy.reshape(1, -1), axis=1)
    return float(np.sum(changed * weights))


def analyse_level(
    specificity: int,
    model_probs: np.ndarray,
    guidance: np.ndarray,
    energy: np.ndarray,
) -> Dict[str, float]:
    sources, x_code, weights, prompts = enumerate_experiment(specificity, model_probs, guidance)
    game = information_game(sources, x_code, weights)
    phis = shapley_values(game)
    h_x = weighted_entropy(x_code, weights)
    efficiency_error = abs(float(phis.sum()) - h_x)
    if efficiency_error > 1e-8:
        raise AssertionError(f"Shapley efficiency error {efficiency_error}")

    v_p = game[(0,)]
    v_pm = game[(0, 1)]
    v_all = game[(0, 1, 2)]
    sequential = np.array((v_p, v_pm - v_p, v_all - v_pm))
    if np.any(sequential < -1e-9):
        raise AssertionError(f"negative chain-rule contribution: {sequential}")
    pair_denom = phis[0] + phis[1]
    return {
        "specificity": int(specificity),
        "prompt_token_count": int(specificity),
        "enumerated_state_count": int(len(weights)),
        "output_entropy_bits": h_x,
        "mi_prompt_bits": v_p,
        "mi_model_given_prompt_bits": float(v_pm - v_p),
        "mi_seed_given_prompt_model_bits": float(v_all - v_pm),
        "chain_prompt_share": float(sequential[0] / h_x),
        "chain_model_share": float(sequential[1] / h_x),
        "chain_seed_share": float(sequential[2] / h_x),
        "shapley_prompt_bits": float(phis[0]),
        "shapley_model_bits": float(phis[1]),
        "shapley_seed_bits": float(phis[2]),
        "shapley_prompt_share": float(phis[0] / h_x),
        "shapley_model_share": float(phis[1] / h_x),
        "shapley_seed_share": float(phis[2] / h_x),
        "shapley_prompt_among_prompt_model": float(phis[0] / pair_denom),
        "attention_prompt_share": expected_attention_share(prompts, guidance, energy),
        "causal_prompt_pixel_change_share": expected_causal_prompt_change(sources, x_code, weights, energy),
        "shapley_efficiency_error_bits": efficiency_error,
    }


def render_example_strip(model_probs: np.ndarray) -> None:
    font = ImageFont.load_default()
    target = np.array([0, 0, 2, 1], dtype=np.int16)  # circle, red, right, stripes
    # Pick a model proposal that visibly conflicts with the target.
    proposal = np.array([2, 1, 0, 2], dtype=np.int16)  # triangle, green, left, dots
    noise_id = 1
    route = np.ones(4, dtype=np.int16)
    prompts = []
    for specificity in range(5):
        p = np.full(4, -1, dtype=np.int16)
        p[:specificity] = target[:specificity]
        prompts.append(p)

    cell_w, cell_h = 154, 110
    canvas = Image.new("RGB", (cell_w * 5, cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, prompt in enumerate(prompts):
        output = proposal.copy()
        for attr in range(4):
            if prompt[attr] >= 0 and route[attr]:
                output[attr] = prompt[attr]
        picture = render_scene(output, noise_id, 72).resize((72, 72))
        x0 = i * cell_w
        canvas.paste(picture, (x0 + 41, 4))
        if i == 0:
            label = "empty prompt"
        else:
            label = ", ".join(ATTR_VALUES[j][prompt[j]] for j in range(4) if prompt[j] >= 0)
        draw.text((x0 + 5, 80), f"{i} token{'s' if i != 1 else ''}", fill=(30, 34, 38), font=font)
        draw.text((x0 + 5, 94), label[:24], fill=(75, 80, 86), font=font)
    canvas.save(ARTIFACTS / "progressive_prompt_examples.png")


def render_prior_grid(rng: np.random.Generator, model_probs: np.ndarray) -> None:
    choices = rng.choice(len(ALL_TUPLES), size=12, p=model_probs)
    tile = 72
    canvas = Image.new("RGB", (tile * 6, tile * 2), "white")
    for idx, choice in enumerate(choices):
        image = render_scene(ALL_TUPLES[choice], idx % 4, tile)
        canvas.paste(image, ((idx % 6) * tile, (idx // 6) * tile))
    canvas.save(ARTIFACTS / "learned_prior_samples.png")


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    xa -= xa.mean()
    ya -= ya.mean()
    return float(np.sum(xa * ya) / math.sqrt(float(np.sum(xa**2) * np.sum(ya**2))))


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def specific_information(
    source_code: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> Dict[int, float]:
    """Compute Williams--Beer specific information I(T=t;S) for each t."""
    source_code = np.asarray(source_code, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    out: Dict[int, float] = {}
    for t in np.unique(target):
        target_mask = target == t
        p_t = float(weights[target_mask].sum())
        value = 0.0
        for s in np.unique(source_code[target_mask]):
            joint_mask = target_mask & (source_code == s)
            p_st = float(weights[joint_mask].sum())
            p_s = float(weights[source_code == s].sum())
            p_s_given_t = p_st / p_t
            p_t_given_s = p_st / p_s
            value += p_s_given_t * math.log2(p_t_given_s / p_t)
        out[int(t)] = value
    return out


def canonical_synergy_benchmark() -> Dict[str, object]:
    """A 256-state stress test with known unique/redundant/synergistic atoms.

    Independent bits are mapped to:
      P=(a,r,c1,c2), M=(b,r,e1,e2), Z=z
      Y=(a,b,r,c1 XOR e1,c2 XOR e2,z)
    """
    base = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.int16)
    a, b, r, c1, c2, e1, e2, z = (base[:, i] for i in range(8))
    p_rows = np.column_stack((a, r, c1, c2))
    m_rows = np.column_stack((b, r, e1, e2))
    y_rows = np.column_stack((a, b, r, c1 ^ e1, c2 ^ e2, z))
    p_code = encode_base(p_rows, 2)
    m_code = encode_base(m_rows, 2)
    z_code = z.astype(np.int64)
    y_code = encode_base(y_rows, 2)
    weights = np.full(len(base), 1.0 / len(base), dtype=float)
    game = information_game((p_code, m_code, z_code), y_code, weights)
    phis = shapley_values(game)
    h_y = weighted_entropy(y_code, weights)

    # Factor-wise Williams--Beer I_min PID. This particular construction has
    # independent output factors, so the factorwise atoms add exactly.
    pid = {"prompt_unique_bits": 0.0, "model_unique_bits": 0.0, "redundant_bits": 0.0, "synergy_bits": 0.0}
    pm_code = p_code + 16 * m_code
    for feature in range(5):  # z is reported separately as seed/noise.
        target = y_rows[:, feature]
        i_p = information_game((p_code, np.zeros_like(p_code), np.zeros_like(p_code)), target, weights)[(0,)]
        i_m = information_game((m_code, np.zeros_like(m_code), np.zeros_like(m_code)), target, weights)[(0,)]
        i_pm = information_game((pm_code, np.zeros_like(pm_code), np.zeros_like(pm_code)), target, weights)[(0,)]
        spec_p = specific_information(p_code, target, weights)
        spec_m = specific_information(m_code, target, weights)
        p_target = {int(t): float(weights[target == t].sum()) for t in np.unique(target)}
        redundancy = sum(p_target[t] * min(spec_p[t], spec_m[t]) for t in p_target)
        unique_p = i_p - redundancy
        unique_m = i_m - redundancy
        synergy = i_pm - unique_p - unique_m - redundancy
        pid["prompt_unique_bits"] += unique_p
        pid["model_unique_bits"] += unique_m
        pid["redundant_bits"] += redundancy
        pid["synergy_bits"] += synergy

    prompt_first = {
        "prompt_bits": game[(0,)],
        "model_given_prompt_bits": game[(0, 1)] - game[(0,)],
        "seed_given_prompt_model_bits": h_y - game[(0, 1)],
    }
    model_first = {
        "model_bits": game[(1,)],
        "prompt_given_model_bits": game[(0, 1)] - game[(1,)],
        "seed_given_model_prompt_bits": h_y - game[(0, 1)],
    }
    pid["seed_noise_bits"] = 1.0
    pid["reconstruction_error_bits"] = abs(
        pid["prompt_unique_bits"]
        + pid["model_unique_bits"]
        + pid["redundant_bits"]
        + pid["synergy_bits"]
        + pid["seed_noise_bits"]
        - h_y
    )
    return {
        "description": "Closed-form 256-state stress test for order, redundancy, and synergy",
        "output_entropy_bits": h_y,
        "coalition_values_bits": {
            "empty": 0.0,
            "prompt": game[(0,)],
            "model": game[(1,)],
            "seed": game[(2,)],
            "prompt_model": game[(0, 1)],
            "prompt_seed": game[(0, 2)],
            "model_seed": game[(1, 2)],
            "all": game[(0, 1, 2)],
        },
        "conditional_mi_prompt_first": prompt_first,
        "conditional_mi_model_first": model_first,
        "pid_i_min_factorwise": pid,
        "shapley_bits": dict(zip(PLAYER_NAMES, map(float, phis))),
        "shapley_shares": dict(zip(PLAYER_NAMES, map(float, phis / h_y))),
        "shapley_efficiency_error_bits": abs(float(phis.sum()) - h_y),
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    counts, model_probs = learn_model_prior(rng)
    true_probs = true_prior_probabilities()
    prior_kl = float(np.sum(true_probs * np.log2(true_probs / model_probs)))
    prior_entropy = float(-np.sum(model_probs * np.log2(model_probs)))
    energy = renderer_feature_energy()

    results = [
        analyse_level(s, model_probs, BASE_GUIDANCE, energy) for s in range(5)
    ]

    guidance_conditions = {
        "weak": np.full(4, 0.60, dtype=float),
        "base": BASE_GUIDANCE,
        "strong": np.array([0.99, 0.98, 0.97, 0.96], dtype=float),
    }
    sensitivity: List[Dict[str, float]] = []
    for name, guidance in guidance_conditions.items():
        row = analyse_level(2, model_probs, guidance, energy)
        sensitivity.append(
            {
                "guidance_condition": name,
                "mean_prompt_attention_probability": float(guidance.mean()),
                "shapley_prompt_share": row["shapley_prompt_share"],
                "shapley_model_share": row["shapley_model_share"],
                "shapley_seed_share": row["shapley_seed_share"],
                "attention_prompt_share": row["attention_prompt_share"],
                "causal_prompt_pixel_change_share": row["causal_prompt_pixel_change_share"],
            }
        )

    attention = [row["attention_prompt_share"] for row in results]
    shapley_pair = [row["shapley_prompt_among_prompt_model"] for row in results]
    causal = [row["causal_prompt_pixel_change_share"] for row in results]
    correlations = {
        "attention_vs_shapley_pair_pearson": pearson(attention, shapley_pair),
        "attention_vs_shapley_pair_spearman": pearson(rankdata(attention), rankdata(shapley_pair)),
        "attention_vs_causal_change_pearson": pearson(attention, causal),
        "attention_vs_causal_change_spearman": pearson(rankdata(attention), rankdata(causal)),
        "attention_vs_causal_change_mae": float(np.mean(np.abs(np.asarray(attention) - np.asarray(causal)))),
    }
    canonical = canonical_synergy_benchmark()

    prior_rows: List[Dict[str, object]] = []
    for attrs, count, learned_p, true_p in zip(ALL_TUPLES, counts, model_probs, true_probs):
        prior_rows.append(
            {
                "shape": ATTR_VALUES[0][attrs[0]],
                "colour": ATTR_VALUES[1][attrs[1]],
                "position": ATTR_VALUES[2][attrs[2]],
                "texture": ATTR_VALUES[3][attrs[3]],
                "training_count": int(count),
                "learned_probability": float(learned_p),
                "true_probability": float(true_p),
            }
        )

    write_csv(ARTIFACTS / "results.csv", results)
    write_csv(ARTIFACTS / "guidance_sensitivity.csv", sensitivity)
    write_csv(ARTIFACTS / "learned_prior.csv", prior_rows)
    render_example_strip(model_probs)
    render_prior_grid(rng, model_probs)

    payload = {
        "experiment": {
            "seed": SEED,
            "training_rows": TRAINING_ROWS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "attribute_names": ATTR_NAMES,
            "attribute_values": ATTR_VALUES,
            "base_guidance": BASE_GUIDANCE.tolist(),
            "renderer_feature_energy": dict(zip(ATTR_NAMES, energy.tolist())),
            "learned_prior_entropy_bits": prior_entropy,
            "true_to_learned_prior_kl_bits": prior_kl,
        },
        "results": results,
        "guidance_sensitivity": sensitivity,
        "correlations": correlations,
        "canonical_synergy_benchmark": canonical,
    }
    results_path = ARTIFACTS / "results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(results_path.read_bytes()).hexdigest()
    (ARTIFACTS / "results.sha256").write_text(
        f"{digest}  results.json\n", encoding="utf-8"
    )

    print(json.dumps(payload, indent=2))
    print(f"\nSHA256 results.json: {digest}")


if __name__ == "__main__":
    main()
