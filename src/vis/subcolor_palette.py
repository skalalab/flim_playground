"""Generate a deterministic family of distinguishable colours from one seed.

Subcolor uses one palette for all distinct values in a figure. Sample colours in
OKLCH, keeping hue and chroma near the seed, then score separation and visibility
after gamut mapping and alpha compositing over the plot background.

Keep this module free of app and scipy imports: standalone script exports inline
it and rely on numpy. Determinism keeps exported colours consistent with the app.
"""

import math

import numpy as np

# Below this chroma, use lightness to distinguish neutral colours.
_NEUTRAL_C = 0.035

# Bound cache growth across Streamlit reruns.
_CACHE_LIMIT = 64
_CACHE = {}


def _srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(np.clip(rgb, 0.0, None), 1.0 / 2.4) - 0.055,
    )


def _linear_rgb_to_oklab(rgb):
    r, g, b = np.moveaxis(np.asarray(rgb, dtype=float), -1, 0)
    l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
    m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
    s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
    lr, mr, sr = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553*lr + 0.7936177850*mr - 0.0040720468*sr,
        1.9779984951*lr - 2.4285922050*mr + 0.4505937099*sr,
        0.0259040371*lr + 0.7827717662*mr - 0.8086757660*sr,
    ], axis=-1)


def _srgb_to_oklab(rgb):
    return _linear_rgb_to_oklab(_srgb_to_linear(rgb))


def _oklab_to_linear_rgb(lab):
    L, a, b = np.moveaxis(np.asarray(lab, dtype=float), -1, 0)
    lr = (L + 0.3963377774*a + 0.2158037573*b) ** 3
    mr = (L - 0.1055613458*a - 0.0638541728*b) ** 3
    sr = (L - 0.0894841775*a - 1.2914855480*b) ** 3
    return np.stack([
        4.0767416621*lr - 3.3077115913*mr + 0.2309699292*sr,
        -1.2684380046*lr + 2.6097574011*mr - 0.3413193965*sr,
        -0.0041960863*lr - 0.7034186147*mr + 1.7076147010*sr,
    ], axis=-1)


def _oklab_to_oklch(lab):
    L, a, b = np.moveaxis(np.asarray(lab, dtype=float), -1, 0)
    return np.stack([L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360], axis=-1)


def _oklch_to_oklab(lch):
    L, C, h = np.moveaxis(np.asarray(lch, dtype=float), -1, 0)
    angle = np.deg2rad(h)
    return np.stack([L, C*np.cos(angle), C*np.sin(angle)], axis=-1)


def _alpha_over(foreground, alpha, background):
    """The colour actually seen when ``foreground`` is drawn at ``alpha`` on ``background``."""
    return alpha*np.asarray(foreground) + (1.0-alpha)*np.asarray(background)


def _gamut_map(lch, iterations=12):
    """Bring OKLCH colours into sRGB by reducing chroma only.

    Binary-search chroma to preserve hue and lightness within the RGB gamut.
    """
    lch = np.asarray(lch, dtype=float)
    L = np.clip(lch[..., 0], 0.0, 1.0)
    wanted = np.clip(lch[..., 1], 0.0, None)
    h = lch[..., 2] % 360.0
    low = np.zeros_like(wanted)
    high = wanted.copy()
    for _ in range(iterations):
        C = (low + high) / 2.0
        linear = _oklab_to_linear_rgb(_oklch_to_oklab(np.stack([L, C, h], axis=-1)))
        valid = np.all((linear >= 0.0) & (linear <= 1.0), axis=-1)
        low = np.where(valid, C, low)
        high = np.where(valid, high, C)
    mapped = np.stack([L, low, h], axis=-1)
    linear = _oklab_to_linear_rgb(_oklch_to_oklab(mapped))
    return _linear_to_srgb(np.clip(linear, 0.0, 1.0)), mapped


def _overlap_confusion(rgb, shown_lab, alpha, background):
    """Return the smallest distance from a two-colour overlap to a third entry.

    Compare alpha-composited colours to detect overlaps that resemble another
    palette member. Return ``inf`` when fewer than three entries exist.
    """
    count, n, _ = rgb.shape
    if n < 3:
        return np.full(count, np.inf)
    best = np.full(count, np.inf)
    for top in range(n):
        for lower in range(n):
            if top == lower:
                continue
            mixed = _alpha_over(rgb[:, top], alpha, _alpha_over(rgb[:, lower], alpha, background))
            mixed_lab = _srgb_to_oklab(mixed)
            other = [index for index in range(n) if index not in (top, lower)]
            distance = np.linalg.norm(mixed_lab[:, None] - shown_lab[:, other], axis=2).min(axis=1)
            best = np.minimum(best, distance)
    return best


def _sample_family(rng, position, anchor, seed_lch, neutral, n, samples):
    """Sample (L, C, h) grids around the seed at ``anchor``.

    One ordered progression drives hue and lightness together. Hue spans at most
    145 degrees, chroma stays near the seed, and lightness uses the available
    range on each side. Hue and lightness ranges grow with ``n`` to separate
    larger palettes; neutral seeds vary mainly in lightness.

    Lightness bounds assume a light plot background. The supplied background
    only affects scoring, including the visibility penalty.
    """
    # Perturb then sort the progression to search perceptual spacing while retaining order.
    t = np.broadcast_to(position, (samples, n)).copy()
    t += rng.normal(0.0, max(0.015, 0.045/math.sqrt(max(1.0, n/3))), (samples, n))
    t = np.clip(t, 0.0, 1.0)
    t.sort(axis=1)
    rows = np.arange(samples)
    t_seed = t[rows, anchor][:, None]
    offset = t - t_seed
    # Guard division when the seed occupies an endpoint.
    below = np.maximum(t_seed - t[:, :1], 1e-6)
    above = np.maximum(t[:, -1:] - t_seed, 1e-6)

    # Lightness reach per side, grown with n and clamped to the room the seed leaves.
    wanted = min(0.42, 0.075 + 0.050*max(0, n-2))
    if neutral:
        # Neutral palettes need a wider lightness range, including for seeds near an endpoint.
        wanted = min(0.70, 0.10 + 0.085*max(0, n-1))
    # Keep hues above near-black unless the seed itself is darker.
    floor = min(0.30, float(seed_lch[0]))
    scale = rng.uniform(0.70, 1.0, (samples, 1))
    down = np.minimum(wanted*scale, max(seed_lch[0]-floor, 0.0))
    # Limit lightness to retain visibility on white and chroma within the sRGB gamut.
    up = np.minimum(wanted*scale, max(0.88 - 0.35*seed_lch[1] - seed_lch[0], 0.0))
    # Keep lightness monotone through the seed and exact at its anchor.
    L = seed_lch[0] + np.where(offset < 0.0, offset/below*down, offset/above*up)
    L = np.clip(L, min(0.24, float(seed_lch[0])), max(0.88, float(seed_lch[0])))

    if neutral:
        # Keep neutral entries at the seed hue to avoid varying colour casts.
        h = np.full((samples, n), seed_lch[2])
        C = np.clip(rng.normal(seed_lch[1], 0.006, (samples, n)), 0.0, _NEUTRAL_C)
    else:
        # Widen the hue arc with palette size while keeping small families close.
        cap = min(145.0, 58.0 + 16.0*n)
        span = rng.uniform(0.60*cap, cap, (samples, 1))
        h = (seed_lch[2] + offset*span) % 360.0
        C = np.clip(rng.normal(seed_lch[1], 0.016, (samples, n)),
                    max(0.020, seed_lch[1]-0.042), seed_lch[1]+0.042)
        # Use slightly more chroma below the seed's lightness and less above it.
        C *= np.clip(1.0 - 0.45*(L-seed_lch[0]), 0.85, 1.15)
        # Taper chroma for very light entries.
        C *= np.clip(1.0 - 0.65*np.maximum(L-0.78, 0.0), 0.74, 1.0)
    return L, C, h


def make_palette(seed_rgb, n, alpha=0.7, background=(1.0, 1.0, 1.0),
                 samples=6000, rng_seed=20260820):
    """Return ``n`` distinguishable RGB float triples, with ``seed_rgb`` first.

    ``seed_rgb`` contains RGB values in 0..1; ``n`` is coerced to at least one.
    ``alpha`` and ``background`` define the displayed colours used for scoring.
    ``samples`` sets the candidate count; ``rng_seed`` makes the search repeatable.
    Score whole palettes by pair separation, visibility, and overlap confusion
    because gamut mapping and compositing change the spacing between entries.

    Vary the seed's position within each candidate, then move it to the front
    of the result without altering its RGB values.
    """
    seed_rgb = np.asarray(seed_rgb, dtype=float).reshape(3)
    n = max(int(n), 1)
    if n == 1:
        return [tuple(float(v) for v in seed_rgb)]

    background = np.asarray(background, dtype=float)
    background_lab = _srgb_to_oklab(background)
    seed_lch = _oklab_to_oklch(_srgb_to_oklab(seed_rgb))
    shown_seed = _srgb_to_oklab(_alpha_over(seed_rgb, alpha, background))
    neutral = bool(seed_lch[1] < _NEUTRAL_C)
    rng = np.random.default_rng(rng_seed)

    position = np.linspace(0.0, 1.0, n)
    anchor = rng.integers(0, n, samples)

    L, C, h = _sample_family(rng, position, anchor, seed_lch, neutral, n, samples)

    h[np.arange(samples), anchor] = seed_lch[2]
    C[np.arange(samples), anchor] = seed_lch[1]
    rgb, mapped = _gamut_map(np.stack([L, C, h], axis=2))
    # Restore the exact seed after the approximate gamut search.
    rgb[np.arange(samples), anchor] = seed_rgb
    mapped[np.arange(samples), anchor] = seed_lch

    shown = _srgb_to_oklab(
        _alpha_over(rgb, alpha, background).reshape(-1, 3)
    ).reshape(samples, n, 3)
    pair = np.linalg.norm(shown[:, :, None, :] - shown[:, None, :, :], axis=-1)
    upper = np.triu_indices(n, 1)
    pair_values = pair[:, upper[0], upper[1]]
    within = pair_values.min(axis=1)
    background_min = np.linalg.norm(shown-background_lab, axis=2).min(axis=1)
    visibility_target = min(0.080, max(0.030, 0.60*np.linalg.norm(shown_seed-background_lab)))
    # Score closest-pair and mean separation, penalizing low background visibility.
    score = (
        within + 0.12*pair_values.mean(axis=1)
        - 0.75*np.maximum(visibility_target-background_min, 0.0)
    )

    # Limit the more expensive overlap-confusion scoring to the strongest candidates.
    ranked = np.argsort(score)[::-1][:min(samples, 400)]
    overlap = _overlap_confusion(rgb[ranked], shown[ranked], alpha, background)
    effective = np.where(np.isfinite(overlap), np.minimum(overlap, within[ranked]), within[ranked])
    best = int(ranked[np.argmax(score[ranked] + 0.18*effective)])

    chosen_rgb, chosen_lch = rgb[best], mapped[best]
    slot = int(anchor[best])
    others = [index for index in range(n) if index != slot]
    # Return the seed first, then sort by hue around it, or by lightness for neutral seeds.
    if neutral:
        others.sort(key=lambda index: float(chosen_lch[index, 0]))
    else:
        offset = (chosen_lch[:, 2]-seed_lch[2]+180.0) % 360.0 - 180.0
        others.sort(key=lambda index: float(offset[index]))
    return [tuple(float(v) for v in chosen_rgb[index]) for index in [slot, *others]]


def make_palette_cached(seed_rgb, n, alpha=0.7, background=(1.0, 1.0, 1.0),
                        samples=6000, rng_seed=20260820):
    """Memoize ``make_palette`` by its arguments across figure reruns.

    The search is pure and deterministic; a bounded cache avoids repeating it.
    """
    key = (tuple(round(float(v), 9) for v in np.asarray(seed_rgb).reshape(3)),
           int(n), float(alpha), tuple(float(v) for v in background), int(samples),
           int(rng_seed))
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = make_palette(seed_rgb, n, alpha, background, samples, rng_seed)
    return list(_CACHE[key])
