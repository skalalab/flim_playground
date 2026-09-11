"""Random celebratory finishing effects for the extraction page (src/celebrate.py)."""
import re
from itertools import pairwise

import pytest
from streamlit.testing.v1 import AppTest

from src.celebrate import (
    BALLOONS,
    BUBBLES,
    CELEBRATION_CLASS,
    CHERRY_BLOSSOMS,
    CONFETTI,
    CSS_EFFECT_BUILDERS,
    EFFECT_NAMES,
    EMOJI_RAIN,
    FIREWORKS,
    SHOOTING_STARS,
    choose_effect,
    render_effect,
)
from src.emojis import happy_celebratory_emojis


def test_pool_holds_several_distinct_effects():
    assert {"shooting stars", "cherry blossoms", "rocket launch", "landing plane", "arriving train"} <= set(EFFECT_NAMES)
    assert "fireflies" not in EFFECT_NAMES
    assert "photon party" not in EFFECT_NAMES
    assert len(set(EFFECT_NAMES)) == len(EFFECT_NAMES)


def test_every_pooled_name_has_a_renderer():
    assert set(EFFECT_NAMES) == {BALLOONS} | set(CSS_EFFECT_BUILDERS)


def test_choose_effect_returns_a_pooled_name():
    assert choose_effect(None) in EFFECT_NAMES


def test_choose_effect_never_repeats_the_previous_one():
    for previous in EFFECT_NAMES:
        drawn = {choose_effect(previous) for _ in range(200)}
        assert previous not in drawn


def test_choose_effect_can_still_draw_every_other_effect():
    previous = EFFECT_NAMES[0]
    drawn = {choose_effect(previous) for _ in range(200)}
    assert drawn == set(EFFECT_NAMES) - {previous}


def test_choose_effect_ignores_an_unknown_previous_effect():
    drawn = {choose_effect("retired-effect") for _ in range(200)}
    assert drawn == set(EFFECT_NAMES)


def test_choose_effect_returns_the_only_effect_in_a_single_effect_pool():
    assert choose_effect("solo", names=("solo",)) == "solo"


@pytest.mark.parametrize("name", sorted(CSS_EFFECT_BUILDERS))
def test_css_effect_overlays_the_viewport_without_blocking_clicks(name):
    html = CSS_EFFECT_BUILDERS[name]()
    assert "position: fixed" in html
    assert "pointer-events: none" in html


@pytest.mark.parametrize("name", sorted(CSS_EFFECT_BUILDERS))
def test_css_effect_classes_are_namespaced(name):
    html = CSS_EFFECT_BUILDERS[name]()
    classes = re.findall(r'class="([^"]+)"', html)
    assert classes
    for value in classes:
        for css_class in value.split():
            assert css_class.startswith("fp-"), css_class


@pytest.mark.parametrize("name", sorted(CSS_EFFECT_BUILDERS))
def test_css_effect_settles_invisible_instead_of_looping(name):
    html = CSS_EFFECT_BUILDERS[name]()
    assert "animation-fill-mode: forwards" in html
    assert "infinite" not in html


@pytest.mark.parametrize("name", sorted(CSS_EFFECT_BUILDERS))
def test_css_effect_particles_are_individually_staggered(name):
    html = CSS_EFFECT_BUILDERS[name]()
    delays = set(re.findall(r"animation-delay:\s*([\d.]+)s", html))
    assert len(delays) > 1


# Explicit timings include the parent timings inherited by decorative pseudo-elements.
_TIMED_RULE = re.compile(r"\{([^{}]*animation-duration:\s*[\d.]+s[^{}]*)\}")

_TOTAL_SECONDS = {
    CONFETTI: 4.5,
    FIREWORKS: 4.0,
    BUBBLES: 4.0,
    EMOJI_RAIN: 4.5,
    SHOOTING_STARS: 4.0,
    CHERRY_BLOSSOMS: 3.5,
    "rocket launch": 4.0,
    "landing plane": 4.8,
    "arriving train": 5.0,
}


def _particle_timings(html):
    """(duration, delay) in seconds for every particle the effect animates."""
    out = []
    for body in _TIMED_RULE.findall(html):
        duration = re.search(r"animation-duration:\s*([\d.]+)s", body)
        delay = re.search(r"animation-delay:\s*([\d.]+)s", body)
        out.append((
            float(duration.group(1)),
            float(delay.group(1)) if delay else 0.0,
        ))
    return out


@pytest.mark.parametrize("name", sorted(CSS_EFFECT_BUILDERS))
def test_css_effect_clears_within_the_runtime_budget(name):
    timings = _particle_timings(CSS_EFFECT_BUILDERS[name]())
    assert timings
    worst = max(duration + delay for duration, delay in timings)
    if name == CHERRY_BLOSSOMS:
        assert worst <= _TOTAL_SECONDS[name]
    else:
        assert worst == pytest.approx(_TOTAL_SECONDS[name]), name
    if name == EMOJI_RAIN:
        assert all(duration >= 3.5 for duration, _ in timings)


def test_render_effect_rejects_an_unknown_effect():
    with pytest.raises(ValueError, match="unknown"):
        render_effect("not-an-effect")


@pytest.mark.parametrize("name", sorted(EFFECT_NAMES))
def test_render_effect_plays_every_pooled_effect_in_a_real_app(name):
    app = AppTest.from_string(
        "from src.celebrate import render_effect\n"
        f"render_effect({name!r})\n"
    )
    app.run()
    assert not app.exception


def test_celebrate_remembers_its_pick_and_avoids_an_immediate_repeat():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.celebrate import celebrate\n"
        "st.session_state.setdefault('picks', [])\n"
        "st.session_state['picks'].append(celebrate())\n"
    )
    app.run()
    for _ in range(12):
        app.run()
    picks = app.session_state["picks"]
    assert len(picks) == 13
    assert not app.exception
    assert all(a != b for a, b in pairwise(picks))
    assert len(set(picks)) > 1


# Per-spark rules: burst index, spark index, and that spark's declarations.
_SPARK_RULE = re.compile(r"span:nth-child\((\d+)\) i:nth-child\((\d+)\)\{([^}]*)\}")


def _sparks(html):
    """Parse per-spark rules into (burst, spark, angle, travel scale, colour, delay)."""
    out = []
    for burst, spark, body in _SPARK_RULE.findall(html):
        angle = re.search(r"rotate\((-?[\d.]+)deg\)", body)
        scale = re.search(r"scale\(([\d.]+)\)", body)
        colour = re.search(r"color:\s*(#[0-9a-fA-F]+)", body)
        delay = re.search(r"animation-delay:\s*([\d.]+)s", body)
        out.append((
            int(burst),
            int(spark),
            float(angle.group(1)) if angle else None,
            float(scale.group(1)) if scale else None,
            colour.group(1) if colour else None,
            float(delay.group(1)) if delay else None,
        ))
    return out


def test_fireworks_sparks_travel_different_distances():
    """One travel distance for every spark draws a compass circle, not a burst."""
    sparks = _sparks(CSS_EFFECT_BUILDERS[FIREWORKS]())
    assert sparks
    scales = {spark[3] for spark in sparks}
    assert None not in scales
    assert len(scales) > 1


def test_fireworks_scatters_many_burst_spots():
    """A handful of lonely pops looked sparse; the sky should have several going off."""
    html = CSS_EFFECT_BUILDERS[FIREWORKS]()
    bursts = re.findall(r"<span>(.*?)</span>", html)
    assert len(bursts) >= 9
    origins = set(re.findall(r"span:nth-child\(\d+\)\{left: ([\d.]+)%;top: ([\d.]+)%", html))
    assert len(origins) == len(bursts)


def test_fireworks_sparks_are_not_evenly_spaced_within_a_burst():
    sparks = _sparks(CSS_EFFECT_BUILDERS[FIREWORKS]())
    assert sparks
    first_burst = sorted(s[2] for s in sparks if s[0] == 1)
    assert len(first_burst) > 2
    gaps = {round(b - a, 1) for a, b in pairwise(first_burst)}
    assert len(gaps) > 1


# Per-bubble rules: bubble index and that bubble's declarations.
_BUBBLE_RULE = re.compile(r"\.fp-bubbles i:nth-child\((\d+)\)\{([^}]*)\}")


def _bubbles(html):
    """Parse per-bubble rules into (index, width, height)."""
    out = []
    for index, body in _BUBBLE_RULE.findall(html):
        width = re.search(r"width:\s*([\d.]+)px", body)
        height = re.search(r"height:\s*([\d.]+)px", body)
        out.append((
            int(index),
            float(width.group(1)) if width else None,
            float(height.group(1)) if height else None,
        ))
    return out


def test_bubbles_rise_at_a_gentle_varied_speed():
    """A short celebration should not force bubbles to race across the entire screen."""
    html = CSS_EFFECT_BUILDERS[BUBBLES]()
    speeds = []
    for _, body in _BUBBLE_RULE.findall(html):
        rise = re.search(r"--fp-rise:\s*(-?[\d.]+)px", body)
        duration = re.search(r"animation-duration:\s*([\d.]+)s", body)
        assert rise and duration
        speeds.append(-float(rise.group(1)) / float(duration.group(1)))
    assert speeds
    assert all(25 <= speed <= 85 for speed in speeds), speeds
    assert max(speeds) - min(speeds) > 10


def test_bubbles_are_round_and_translucent():
    """A bubble you cannot see through is just a dot."""
    html = CSS_EFFECT_BUILDERS[BUBBLES]()
    assert "border-radius: 50%" in html
    alphas = [float(a) for a in re.findall(r"rgba\([^)]*,\s*([\d.]+)\s*\)", html)]
    assert alphas
    assert all(0 < alpha < 1 for alpha in alphas), alphas


def test_bubbles_vary_in_size_and_stay_circular():
    bubbles = _bubbles(CSS_EFFECT_BUILDERS[BUBBLES]())
    assert len(bubbles) > 10
    assert len({bubble[1] for bubble in bubbles}) > 1
    assert all(bubble[1] == bubble[2] for bubble in bubbles), bubbles


def test_bubbles_sway_sideways_as_they_rise():
    """Bubbles drift either way independently of their steady ascent."""
    html = CSS_EFFECT_BUILDERS[BUBBLES]()
    drifts = [float(v) for v in re.findall(r"--fp-sway:\s*(-?[\d.]+)px", html)]
    assert any(drift < 0 for drift in drifts)
    assert any(drift > 0 for drift in drifts)
    assert all(abs(drift) <= 24 for drift in drifts)


def test_emoji_rain_can_exceed_the_happy_emoji_pool():
    """A larger shower must reach its count while retaining the full emoji variety."""
    emojis = re.findall(r"<i>(.+?)</i>", CSS_EFFECT_BUILDERS[EMOJI_RAIN](count=100))
    assert len(emojis) == 100
    assert set(emojis) == set(happy_celebratory_emojis)


def test_css_effects_take_their_element_container_out_of_flow():
    """A zero-height overlay still costs the vertical block one gap, shifting later elements."""
    rule = re.compile(
        r'\[data-testid="stElementContainer"\]:has\(\.' + CELEBRATION_CLASS + r'\)\s*\{[^}]*'
        r"position:\s*absolute",
    )
    for name, build in CSS_EFFECT_BUILDERS.items():
        html = build()
        assert f'class="{CELEBRATION_CLASS} ' in html, name
        assert rule.search(html), name
