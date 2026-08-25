"""Generate ``n`` distinguishable colours from one seed colour.

Used by subcolor (``create_subcolor_map`` in src/vis/helpers.py),
which maps every distinct value of the nested column to one colour for the whole
figure. One seed grows the whole palette, sized to the number of distinct values.

Why generate rather than slice a named palette. Slicing fails two ways that matter
here: cyclic palettes (``husl``, ``hls``) put their first and last entries at adjacent
hues, so "take the ends" yields two colours a viewer cannot tell apart; and seaborn
cycles qualitative palettes past their length, so ``tab10`` at 20 colours contains
exact duplicates. Generating for the requested count sidesteps both.

Colours are sampled in OKLCH and scored on how they look *composited over the plot
background* at the opacity the points are drawn with, because scatter points are
semi-transparent and two colours that differ in RGB can composite to nearly the same
thing.

The palette is one *harmonious family*: hue is confined to an arc around the seed,
chroma stays near the seed's own, and lightness steps outward from it, so members look
related rather than merely different. Within that arc the scorer still maximises
separation, so what comes back is the most distinguishable palette that coherence allows.

This was chosen over the alternative -- fanning hue most of the way round the wheel and
shuffling a lightness ramp, which buys roughly 2-3x the separation -- after rendering
both on real sina plots. Two findings decided it. At the group sizes this encoding is
actually used at (2-5 members) the wide fan is no more readable, and it emits entries
that read as mistakes: a near-black, or a pale yellow-green that half-disappears against
white. At 7+ members the wide fan does separate better, but by then neither palette lets
a reader assign a point to a member, so the advantage is moot. See the ``_sample_family``
docstring for what the arc costs and why the bounds sit where they do.

This module is deliberately free of app imports and of scipy: it is inlined verbatim
into exported standalone scripts (src/export_script.py), so it must run with numpy
alone. Everything is deterministic given its arguments, which is what keeps an exported
script's colours identical to the screen's.
"""

import math

import numpy as np

# Chroma below which a seed is treated as a grey. A grey has no meaningful hue to fan
# around, so its palette is built from lightness instead.
_NEUTRAL_C = 0.035

# Bound on the memo. A Streamlit process lives for days and each (seed, count) the user
# lands on adds one entry.
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

    Clipping RGB instead would shift hue and lightness, which is what distinguishes the
    entries from each other; binary-searching chroma keeps both and only desaturates
    until the colour is representable.
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
    """Closest approach between a two-deep overlap and some *third* palette entry.

    Semi-transparent points pile up, and two stacked entries composite into a new
    colour. Where that stack lands on a third entry's colour, a dense region reads as
    the wrong member. Returns ``inf`` below 3 entries, where no third exists.
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
    """Candidate (L, C, h) grids: one coherent family rather than unrelated colours.

    One progression variable ``t`` drives hue and lightness together, so a step along the
    palette advances both at once -- co-varying them is most of what makes a set of
    colours read as a family rather than as a list. The seed sits at slot ``anchor`` and
    is the fixed point of both: offsets are measured from *its* position in ``t``, so the
    seed is the colour the family grew out of rather than merely one of its members.

    Three things are deliberately narrow, and each is a bound the scorer would happily
    blow past if it were left free -- separation is all it optimises, and it will spend
    any coherence it is given:

    hue        a sampled arc of at most ~145 deg rather than most of the wheel. Wider
               arcs separate better and stop looking related; this is as far open as the
               family still reads as one.
    chroma     held within a hair of the seed's own, so entries share one saturation
               character. A palette mixing a pastel with a fully saturated neighbour
               reads as two palettes however close their hues.
    lightness  travels outward from the seed instead of ramping across a shuffled range,
               and the travel is bounded by the room actually available on each side, so
               a dark seed leans upward and a light one downward rather than clipping.
               These bounds assume the light plot background the app draws on -- the
               sampler is not passed ``background``, so on a dark one the darkest member
               can approach it. Only the scorer sees the background, and only through its
               visibility term.

    Both the arc and the lightness travel grow with ``n``, because neither alone can
    separate many entries: at eight members even a 145 deg arc leaves ~21 deg between
    neighbours, which is under the confusion threshold once alpha compositing has shrunk
    it. So the family opens up on both axes as it gets more crowded, and a pair of
    members -- the common case here -- stays tight.
    """
    # Shared progression. Noise then sort, so entries stay ordered along the family while
    # not sitting on a perfectly even grid -- an even grid is a worse starting point for
    # the scorer, which is trying to even out *perceived* rather than nominal spacing.
    t = np.broadcast_to(position, (samples, n)).copy()
    t += rng.normal(0.0, max(0.015, 0.045/math.sqrt(max(1.0, n/3))), (samples, n))
    t = np.clip(t, 0.0, 1.0)
    t.sort(axis=1)
    rows = np.arange(samples)
    t_seed = t[rows, anchor][:, None]
    offset = t - t_seed
    # How much of the progression falls on each side of the seed. Guard the division: the
    # seed can land on either end, leaving one side empty.
    below = np.maximum(t_seed - t[:, :1], 1e-6)
    above = np.maximum(t[:, -1:] - t_seed, 1e-6)

    # Lightness reach per side, grown with n and clamped to the room the seed leaves.
    wanted = min(0.42, 0.075 + 0.050*max(0, n-2))
    if neutral:
        # A grey has neither hue nor chroma to vary, so lightness is carrying the whole
        # palette and gets the full room rather than a share of it. The ceiling is the
        # usable range itself, not a fixed reach: a seed sitting near black or near white
        # can travel in one direction only, and a reach tuned for a mid grey would crowd
        # every member into a fraction of the range.
        wanted = min(0.70, 0.10 + 0.085*max(0, n-1))
    # A hair off black stops reading as a hue at all, which costs the family a member;
    # the floor gives way to a seed that is already darker than it rather than dragging
    # the seed's own family upward away from it.
    floor = min(0.30, float(seed_lch[0]))
    scale = rng.uniform(0.70, 1.0, (samples, 1))
    down = np.minimum(wanted*scale, max(seed_lch[0]-floor, 0.0))
    # Ceiling well short of white, and lower the more saturated the seed is. Two separate
    # things go wrong at the top of the range. A semi-transparent point at L above ~0.9
    # composited over a white background lands within the confusion threshold *of the
    # background*, so the member reads as absent rather than as faint. And sRGB simply
    # holds less chroma the lighter a colour gets, so a saturated seed's family loses its
    # saturation on the way up -- climbing to 0.88 turned a 0.165-chroma pink into a
    # 0.063-chroma wash, which is the same washed-out entry by another route. Ceding the
    # top of the range costs nothing, because the travel goes downward instead, and dark
    # is where the gamut has chroma to spare.
    up = np.minimum(wanted*scale, max(0.88 - 0.35*seed_lch[1] - seed_lch[0], 0.0))
    # Piecewise linear in ``offset`` and increasing on both sides, so the ramp is monotone
    # through the seed and lands exactly on the seed's own lightness at ``anchor``.
    L = seed_lch[0] + np.where(offset < 0.0, offset/below*down, offset/above*up)
    L = np.clip(L, min(0.24, float(seed_lch[0])), max(0.88, float(seed_lch[0])))

    if neutral:
        # Hue is meaningless at this chroma; keep it at the seed's so the family cannot
        # pick up a faint cast that differs entry to entry.
        h = np.full((samples, n), seed_lch[2])
        C = np.clip(rng.normal(seed_lch[1], 0.006, (samples, n)), 0.0, _NEUTRAL_C)
    else:
        # The arc opens with n rather than being one width for every size. The scorer
        # always spends whatever arc it is given, so a fixed cap would make a pair as far
        # apart as an octet needs to be -- and two colours 110 deg apart are trivially
        # distinguishable while no longer looking related, which is separation bought at
        # the exact price this palette exists to avoid paying.
        cap = min(145.0, 58.0 + 16.0*n)
        span = rng.uniform(0.60*cap, cap, (samples, 1))
        h = (seed_lch[2] + offset*span) % 360.0
        C = np.clip(rng.normal(seed_lch[1], 0.016, (samples, n)),
                    max(0.020, seed_lch[1]-0.042), seed_lch[1]+0.042)
        # Tie chroma gently to lightness, the way a hand-picked family does it: the
        # entries below the seed carry a little more, the ones above a little less. It
        # reads as one family lit from one side instead of a row of equally saturated
        # chips, and it buys separation that a muted seed cannot get from hue alone --
        # without it a low-chroma seed like Set2 turns to mud once crowded.
        C *= np.clip(1.0 - 0.45*(L-seed_lch[0]), 0.85, 1.15)
        # A very light colour held at full chroma goes garish and loses its edge against
        # a pale background, so taper chroma as lightness climbs.
        C *= np.clip(1.0 - 0.65*np.maximum(L-0.78, 0.0), 0.74, 1.0)
    return L, C, h


def make_palette(seed_rgb, n, alpha=0.7, background=(1.0, 1.0, 1.0),
                 samples=6000, rng_seed=20260820):
    """Return ``n`` distinguishable (r, g, b) tuples seeded by ``seed_rgb``.

    seed_rgb   : (r, g, b) floats in 0..1 — always returned first, bit-for-bit
    n          : how many colours to produce (>= 1)
    alpha      : the opacity the points are drawn at, so entries are compared as seen
    background : plot background the points are composited over
    samples    : candidate palettes to draw; the best-scoring one wins
    rng_seed   : fixes the draw, so the same arguments always give the same palette

    Whole palettes are sampled and then scored, rather than each colour being placed by
    a rule, because the thing being optimised — the *smallest* gap between any two
    entries after gamut mapping and alpha compositing — is a property of the set, not of
    any one colour. Even hue spacing in OKLCH is not even once those two steps have run.

    The seed occupies a random slot per sample rather than always the first, so the
    palette is not forced to treat the seed as its darkest or its middle entry; it is
    moved to the front only on the way out.
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
    # Restore the seed exactly: the gamut map's binary search leaves a sub-1/255 residue,
    # and the seed must round-trip bit-for-bit so the first entry equals the colour the
    # caller asked to build around.
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
    # Maximise the closest pair first — one confusable pair spoils the palette however
    # well separated the rest are — then the mean as a tie-break, and penalise any entry
    # fading into the background.
    score = (
        within + 0.12*pair_values.mean(axis=1)
        - 0.75*np.maximum(visibility_target-background_min, 0.0)
    )

    # Re-score only the strongest candidates for overlap confusion: it is O(n^2) per
    # candidate, and a palette that already loses on the cheap terms cannot win on this.
    ranked = np.argsort(score)[::-1][:min(samples, 400)]
    overlap = _overlap_confusion(rgb[ranked], shown[ranked], alpha, background)
    effective = np.where(np.isfinite(overlap), np.minimum(overlap, within[ranked]), within[ranked])
    best = int(ranked[np.argmax(score[ranked] + 0.18*effective)])

    chosen_rgb, chosen_lch = rgb[best], mapped[best]
    slot = int(anchor[best])
    others = [index for index in range(n) if index != slot]
    # Seed first, then the rest in a stable perceptual order — by hue around the seed,
    # or by lightness for a grey seed — so consecutive entries are neighbours and a
    # legend read top to bottom walks the palette rather than jumping around it.
    if neutral:
        others.sort(key=lambda index: float(chosen_lch[index, 0]))
    else:
        offset = (chosen_lch[:, 2]-seed_lch[2]+180.0) % 360.0 - 180.0
        others.sort(key=lambda index: float(offset[index]))
    return [tuple(float(v) for v in chosen_rgb[index]) for index in [slot, *others]]


def make_palette_cached(seed_rgb, n, alpha=0.7, background=(1.0, 1.0, 1.0),
                        samples=6000, rng_seed=20260820):
    """:func:`make_palette`, memoised on its arguments.

    The search costs a fraction of a second (measured: 25 ms at n=2, 232 ms at n=20,
    7 us on a hit) and there is one call per figure, repeated on every Streamlit rerun --
    which is what the memo is for. Safe because the search is pure and deterministic.
    """
    key = (tuple(round(float(v), 9) for v in np.asarray(seed_rgb).reshape(3)),
           int(n), float(alpha), tuple(float(v) for v in background), int(samples),
           int(rng_seed))
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = make_palette(seed_rgb, n, alpha, background, samples, rng_seed)
    return list(_CACHE[key])
