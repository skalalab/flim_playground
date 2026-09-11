"""Random celebratory effects played once when a long job finishes.

Streamlit ships only two full-screen effects and ``st.snow`` reads as winter
rather than celebration, so the rest use CSS and locally bundled artwork. ``st.html`` is not
iframed, so a fixed-position overlay covers the viewport the way
``st.balloons`` does; no JavaScript is needed and none is enabled.

Particle styling lives in generated ``nth-child`` rules rather than inline
``style`` attributes, so nothing depends on how the sanitizer treats them.
"""
import random
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from src.emojis import happy_celebratory_emojis

BALLOONS = "balloons"
CONFETTI = "confetti"
FIREWORKS = "fireworks"
BUBBLES = "bubbles"
EMOJI_RAIN = "emoji rain"
SHOOTING_STARS = "shooting stars"
CHERRY_BLOSSOMS = "cherry blossoms"
ROCKET_LAUNCH = "rocket launch"
LANDING_PLANE = "landing plane"
ARRIVING_TRAIN = "arriving train"

LAST_EFFECT_KEY = "last_celebration_effect"

CELEBRATION_CLASS = "fp-celebration"

# Every custom effect shares these rules. The overlay sits above Streamlit's own chrome
# without competing with modal dialogs; taking its element container out of flow stops
# the vertical block from spending a 16px gap on it, which would shift the elements
# rendered after the effect down for as long as it plays.
_SHARED_RULES = f"""
[data-testid="stElementContainer"]:has(.{CELEBRATION_CLASS}) {{ position: absolute; }}
.{CELEBRATION_CLASS} {{
    position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 9999990;
}}
@media (prefers-reduced-motion: reduce) {{
    .{CELEBRATION_CLASS} {{ display: none; }}
}}
"""
_PARTY_COLORS = ("#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93", "#ff924c", "#f72585")
_FIREWORK_PALETTES = (
    ("#efaa24", "#ffd477", "#fff0be"),
    ("#ee518d", "#ff97bd", "#ffdbeb"),
    ("#26a9ce", "#7bdef0", "#d5faff"),
    ("#9563e6", "#c4a2ff", "#efe1ff"),
)
# Deduplicate the shared happy pool so every emoji has equal weight.
# Larger showers reuse it in shuffled passes to keep the variety balanced.
_RAIN_EMOJIS = tuple(dict.fromkeys(happy_celebratory_emojis))


def _style(rules):
    """Wrap generated CSS rules in a style tag."""
    return "<style>" + "".join(rules) + "</style>"


@lru_cache(maxsize=None)
def _celebration_asset(name):
    """Read a bundled asset; celebrations never fetch artwork at runtime."""
    return (Path(__file__).with_name("celebration_assets") / name).read_text(encoding="utf-8")


def _confetti_html(count=180):
    """Colored slips tumbling down the viewport over 4.5 seconds."""
    rules = [
        _SHARED_RULES,
        """.fp-confetti i {
            position: absolute; top: -12vh; display: block; border-radius: 1px;
            animation-timing-function: linear;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        @keyframes fp-confetti-a {
            0% { transform: translateX(0) rotate(0deg); opacity: 1; }
            80% { opacity: 1; }
            100% { transform: translateX(6vw) translateY(115vh) rotate(760deg); opacity: 0; }
        }
        @keyframes fp-confetti-b {
            0% { transform: translateX(0) rotate(0deg); opacity: 1; }
            80% { opacity: 1; }
            100% { transform: translateX(-7vw) translateY(115vh) rotate(-620deg); opacity: 0; }
        }
        @keyframes fp-confetti-c {
            0% { transform: translateX(0) rotate(0deg); opacity: 1; }
            80% { opacity: 1; }
            100% { transform: translateX(1vw) translateY(115vh) rotate(420deg); opacity: 0; }
        }""",
    ]
    for i in range(count):
        delay = (i % 20) * 0.04
        duration = 4.5 - delay if i == count - 1 else random.uniform(3.0, 3.7)
        rules.append(
            f".fp-confetti i:nth-child({i + 1}){{"
            f"left: {random.uniform(1, 99):.2f}%;"
            f"width: {random.uniform(6, 11):.1f}px;"
            f"height: {random.uniform(9, 17):.1f}px;"
            f"background: {random.choice(_PARTY_COLORS)};"
            f"animation-name: fp-confetti-{'abc'[i % 3]};"
            f"animation-duration: {duration:.2f}s;"
            f"animation-delay: {delay:.2f}s;"
            "}"
        )
    return _style(rules) + f'<div class="{CELEBRATION_CLASS} fp-confetti">' + "<i></i>" * count + "</div>"


def _fireworks_html(bursts=11, dots=28):
    """Launched shells opening into glowing trails, then falling and burning out."""
    rules = [
        _SHARED_RULES,
        """.fp-fireworks span {
            position: absolute; width: 0; height: 0;
            animation-name: fp-firework-fall;
            animation-timing-function: linear;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        .fp-fireworks span::before, .fp-fireworks span::after {
            content: ""; position: absolute; opacity: 0; pointer-events: none;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        .fp-fireworks span::before {
            width: 18px; height: 18px; left: -9px; top: -9px; border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.98),
                currentColor 25%, transparent 70%);
            animation-name: fp-firework-flash;
            animation-timing-function: ease-out;
        }
        .fp-fireworks span::after {
            width: 2px; height: 32px; left: -1px; top: 0;
            border-radius: 50%; transform-origin: top;
            background: linear-gradient(to bottom, rgba(255, 255, 255, 0.95),
                currentColor 20%, transparent);
            filter: drop-shadow(0 0 2px currentColor);
            animation-name: fp-firework-launch;
            animation-timing-function: ease-out;
        }
        .fp-fireworks i { position: absolute; left: 0; top: 0; display: block; }
        .fp-fireworks b {
            display: block; width: 2px; height: 26px; margin-left: -1px;
            border-radius: 50%; transform-origin: top; opacity: 0;
            background: linear-gradient(to bottom, rgba(255, 255, 255, 0.98),
                currentColor 18%, transparent);
            filter: drop-shadow(0 0 2px currentColor);
            animation-name: fp-spark;
            animation-timing-function: linear;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        @keyframes fp-firework-launch {
            0% { transform: translateY(100px) scaleY(0.5); opacity: 0; }
            18% { opacity: 0.9; }
            80% { opacity: 0.9; }
            100% { transform: translateY(0) scaleY(1); opacity: 0; }
        }
        @keyframes fp-firework-flash {
            0% { transform: scale(0.3); opacity: 0; }
            15% { opacity: 0.95; }
            100% { transform: scale(1.6); opacity: 0; }
        }
        @keyframes fp-firework-fall {
            0% { transform: translateY(0); }
            35% { transform: translateY(4px); }
            70% { transform: translateY(24px); }
            100% { transform: translateY(64px); }
        }
        @keyframes fp-spark {
            0% { transform: translateY(0) scaleY(0.2); opacity: 0; }
            5% { opacity: 1; }
            30% { transform: translateY(-62px) scaleY(1); opacity: 1; }
            60% { transform: translateY(-99px) scaleY(0.62); opacity: 0.9; }
            82% { transform: translateY(-116px) scaleY(0.32); opacity: 0.65; }
            100% { transform: translateY(-128px) scaleY(0.06); opacity: 0; }
        }""",
    ]
    # Spread the volley across the screen without a left-to-right marching order.
    positions = random.sample(range(bursts), bursts)
    for burst in range(bursts):
        selector = f".fp-fireworks span:nth-child({burst + 1})"
        palette = _FIREWORK_PALETTES[burst % len(_FIREWORK_PALETTES)]
        # Extend the volley while keeping each shell's launch and burn speed.
        # The last burst starts at 2.55s and its final ember burns out at 4s.
        delay = 0.28 + 2.27 * burst / max(bursts - 1, 1)
        if burst < bursts - 1:
            delay += random.uniform(0, 0.035)
        rules.append(
            f"{selector}{{"
            f"left: {10 + 80 * (positions[burst] + 0.5) / bursts:.1f}%;"
            f"top: {random.uniform(18, 62):.1f}%;"
            f"color: {palette[0]};"
            "animation-duration: 1.45s;"
            f"animation-delay: {delay:.2f}s;"
            "}"
        )
        rules.append(
            f"{selector}::before{{"
            "animation-duration: 0.32s;"
            f"animation-delay: {delay:.2f}s;"
            "}"
            f"{selector}::after{{"
            "animation-duration: 0.28s;"
            f"animation-delay: {delay - 0.28:.2f}s;"
            "}"
        )
        for dot in range(dots):
            # Short inner embers and longer outer trails fill the shell rather than
            # outlining a ring. Gravity is on the unrotated parent, so all sparks fall.
            angle = dot * 360 / dots + random.uniform(-5, 5)
            travel = random.uniform(0.38, 0.7) if dot % 4 == 0 else random.uniform(0.8, 1.3)
            duration = 1.45 if dot == 0 else random.uniform(1.05, 1.45)
            rules.append(
                f"{selector} i:nth-child({dot + 1}){{"
                f"transform: rotate({angle:.1f}deg) scale({travel:.2f});"
                f"color: {random.choice(palette)};"
                "}"
                f"{selector} i:nth-child({dot + 1}) b{{"
                f"animation-duration: {duration:.2f}s;"
                f"animation-delay: {delay:.2f}s;"
                "}"
            )
    burst_html = "<span>" + "<i><b></b></i>" * dots + "</span>"
    return _style(rules) + f'<div class="{CELEBRATION_CLASS} fp-fireworks">' + burst_html * bursts + "</div>"


def _bubbles_html(count=60):
    """Clear soap bubbles with moving thin-film reflections around their curved rims."""
    rules = [
        _SHARED_RULES,
        """.fp-bubbles i {
            position: absolute; display: block; opacity: 0;
            animation-name: fp-bubble-rise;
            animation-timing-function: linear;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        .fp-bubbles b {
            position: relative; display: block; width: 100%; height: 100%;
            box-sizing: border-box; border-radius: 50%;
            border: 1px solid rgba(255, 255, 255, 0.35);
            background-color: transparent;
            box-shadow: inset 1px 1px 2px rgba(255, 255, 255, 0.65),
                inset -1px -2px 3px rgba(95, 115, 185, 0.15),
                0 0 1px rgba(68, 98, 134, 0.35);
            animation-name: fp-bubble-sway;
            animation-duration: inherit;
            animation-delay: inherit;
            animation-timing-function: cubic-bezier(0.45, 0, 0.55, 1);
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        .fp-bubbles b::before, .fp-bubbles b::after {
            content: ""; position: absolute; inset: -1px; border-radius: inherit;
        }
        .fp-bubbles b::before {
            /* The mask leaves the center clear; the spectrum is reflected on the film. */
            background: conic-gradient(
                rgba(255, 116, 191, 0.65), rgba(184, 145, 255, 0.62),
                rgba(75, 199, 255, 0.6), rgba(102, 236, 194, 0.48),
                rgba(255, 218, 132, 0.6), rgba(255, 151, 172, 0.62),
                rgba(255, 116, 191, 0.65));
            -webkit-mask-image: radial-gradient(circle, transparent 56%,
                rgba(0, 0, 0, 0.2) 62%, rgba(0, 0, 0, 0.75) 68%, #000 71%);
            mask-image: radial-gradient(circle, transparent 56%,
                rgba(0, 0, 0, 0.2) 62%, rgba(0, 0, 0, 0.75) 68%, #000 71%);
            transform: rotate(var(--fp-film-angle));
            animation-name: fp-bubble-iridescence;
            animation-duration: inherit;
            animation-delay: inherit;
            animation-timing-function: ease-in-out;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        .fp-bubbles b::after {
            background: radial-gradient(ellipse at 28% 23%,
                rgba(255, 255, 255, 0.92) 0%, rgba(255, 255, 255, 0.6) 3%, transparent 9%),
                radial-gradient(ellipse at 73% 79%,
                rgba(255, 255, 255, 0.7) 0%, transparent 6%);
        }
        @keyframes fp-bubble-iridescence {
            0% { transform: rotate(var(--fp-film-angle)); }
            100% { transform: rotate(calc(var(--fp-film-angle) + 45deg)); }
        }
        @keyframes fp-bubble-rise {
            0% { transform: translate(0, 0); opacity: 0; }
            16% { opacity: 1; }
            72% { opacity: 1; }
            100% { transform: translate(var(--fp-drift), var(--fp-rise)); opacity: 0; }
        }
        @keyframes fp-bubble-sway {
            0% { transform: translateX(calc(var(--fp-sway) * -0.5)); }
            45% { transform: translateX(var(--fp-sway)); }
            100% { transform: translateX(calc(var(--fp-sway) * -0.6)); }
        }""",
    ]
    for i in range(count):
        size = round(random.uniform(22, 58), 1)
        original_duration = random.uniform(2.4, 3.1)
        duration = 4.0 if i == 0 else round(original_duration + 0.7, 2)
        delay = random.uniform(0, min(0.45, 4.0 - duration))
        # More time aloft means more travel at the same gentle rise speed.
        rise = random.uniform(110, 200) * duration / original_duration
        rules.append(
            f".fp-bubbles i:nth-child({i + 1}){{"
            f"left: {random.uniform(3, 92):.2f}%;"
            f"top: {random.uniform(36, 102):.2f}vh;"
            f"width: {size:.1f}px;"
            f"height: {size:.1f}px;"
            f"--fp-film-angle: {random.uniform(0, 360):.1f}deg;"
            f"--fp-rise: {-rise:.1f}px;"
            f"--fp-drift: {random.uniform(-20, 20):.1f}px;"
            f"--fp-sway: {random.uniform(8, 24) * (-1 if i % 2 else 1):.1f}px;"
            f"animation-duration: {duration:.2f}s;"
            f"animation-delay: {delay:.2f}s;"
            "}"
        )
    return (_style(rules) + f'<div class="{CELEBRATION_CLASS} fp-bubbles" aria-hidden="true">'
            + "<i><b></b></i>" * count + "</div>")


def _shooting_stars_html(count=24):
    """A shower of diagonal shooting stars with luminous heads and tapered tails."""
    rules = [
        _SHARED_RULES,
        """.fp-shooting-stars i {
            position: absolute; display: block; width: 0; height: 0;
        }
        .fp-shooting-stars b {
            position: relative; display: block; width: 4px; height: 4px;
            border-radius: 50%; background: rgba(255, 255, 255, 0.98); opacity: 0;
            box-shadow: 0 0 5px 1px currentColor;
            animation-name: fp-shooting-star;
            animation-timing-function: linear;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        .fp-shooting-stars b::before {
            content: ""; position: absolute; right: 2px; top: 1px;
            width: clamp(48px, 11vw, 150px); height: 2px;
            background: linear-gradient(to right, transparent, currentColor);
            border-radius: 50%;
        }
        .fp-shooting-stars b::after {
            content: ""; position: absolute; inset: -4px;
            background: currentColor;
            clip-path: polygon(50% 0%, 60% 40%, 100% 50%, 60% 60%,
                50% 100%, 40% 60%, 0% 50%, 40% 40%);
        }
        @keyframes fp-shooting-star {
            0% { transform: translateX(0); opacity: 0; }
            10% { opacity: 1; }
            72% { opacity: 1; }
            100% { transform: translateX(min(70vw, 90vh, 780px)); opacity: 0; }
        }""",
    ]
    colors = ("#7ab8ed", "#e2b55c", "#a795e7")
    for i in range(count):
        selector = f".fp-shooting-stars i:nth-child({i + 1})"
        duration = 1.12 if i == count - 1 else random.uniform(0.72, 1.12)
        delay = 0.12 + 2.76 * i / max(count - 1, 1)
        if i < count - 1:
            delay = min(2.88, delay + random.uniform(0, 0.07))
        # Keep the final star in view until it fades, including on short screens.
        last = i == count - 1
        rules.append(
            f"{selector}{{"
            f"left: {random.uniform(-8, 12 if last else 68):.1f}%;"
            f"top: {random.uniform(8, 25 if last else 62):.1f}%;"
            f"transform: rotate({random.uniform(24, 38):.1f}deg);"
            f"color: {colors[i % len(colors)]};"
            "}"
            f"{selector} b{{"
            f"animation-duration: {duration:.2f}s;"
            f"animation-delay: {delay:.2f}s;"
            "}"
        )
    return (_style(rules) + f'<div class="{CELEBRATION_CLASS} fp-shooting-stars" aria-hidden="true">'
            + "<i><b></b></i>" * count + "</div>")


def _cherry_blossoms_html(count=28):
    """Thin, notched sakura petals with faint veins and a gentle flutter."""
    # Flora of China describes sakura petals as obovate and emarginate:
    # broad at the outer end, tapering to the base, with a notched tip.
    # https://www.efloras.org/florataxon.aspx?flora_id=2&taxon_id=200010679
    # These original outlines follow the uneven edges and fine, fan-shaped veins
    # visible in the Science Museum's Somei-yoshino close-up:
    # https://www3.jsf.or.jp/mailmaga/photo/sss10/
    # CSS SVG images survive st.html's sanitizer; inline SVG elements do not.
    outlines = (
        "M23 53 C23 49 17 48 12 44 C4 38 1 29 4 19 "
        "C6 11 13 6 20 6 Q24 6 27 9 L29 12 L31 7 "
        "C38 6 45 14 46 24 C47 34 42 43 35 47 "
        "C31 49 28 51 28 54 Q25 56 23 53Z",
        "M25 54 C24 50 18 48 13 43 C6 36 4 27 6 18 "
        "C8 10 14 5 20 4 Q24 4 26 7 L29 11 L31 5 "
        "C39 6 43 13 44 22 C45 33 39 44 32 49 "
        "L29 54 Q27 56 25 54Z",
        "M26 54 C22 50 15 48 10 42 C4 35 4 26 8 18 "
        "C12 10 21 5 28 7 L31 11 L33 8 "
        "C41 11 45 18 43 28 C42 40 35 47 29 54 Q28 55 26 54Z",
    )
    rules = [
        _SHARED_RULES,
        """.fp-cherry-blossoms i {
            position: absolute; display: block; opacity: 0;
            animation-name: fp-petal-fall;
            animation-timing-function: linear;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        .fp-cherry-blossoms b {
            position: relative; display: block; width: 100%; height: 100%;
            background-size: 100% 100%; background-repeat: no-repeat;
            transform-origin: 50% 65%;
            animation-name: fp-petal-tumble;
            animation-duration: inherit;
            animation-delay: inherit;
            animation-timing-function: ease-in-out;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        @keyframes fp-petal-fall {
            0% { transform: translate(0, 0); opacity: 0; }
            14% { opacity: 0.92; }
            75% { opacity: 0.92; }
            100% { transform: translate(var(--fp-wind), var(--fp-fall)); opacity: 0; }
        }
        @keyframes fp-petal-tumble {
            0% { transform: translateX(0) rotate(var(--fp-tilt)) rotateY(-25deg) rotateX(12deg); }
            35% { transform: translateX(var(--fp-flutter))
                rotate(calc(var(--fp-tilt) + 20deg)) rotateY(30deg) rotateX(-15deg); }
            70% { transform: translateX(calc(var(--fp-flutter) * -0.6))
                rotate(calc(var(--fp-tilt) + 48deg)) rotateY(-40deg) rotateX(18deg); }
            100% { transform: translateX(0)
                rotate(calc(var(--fp-tilt) + 72deg)) rotateY(20deg) rotateX(-8deg); }
        }""",
    ]
    for variant, outline in enumerate(outlines):
        petal = quote(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 56">'
            '<defs><linearGradient id="tissue" x1="0.2" y1="0" x2="0.65" y2="1">'
            '<stop stop-color="#fff5f8"/>'
            '<stop offset="0.65" stop-color="#fce9f0"/>'
            '<stop offset="1" stop-color="#f1b9ce"/>'
            '</linearGradient><linearGradient id="fold">'
            '<stop stop-color="#dab2c5" stop-opacity="0"/>'
            '<stop offset="0.48" stop-color="#dab2c5" stop-opacity="0.3"/>'
            '<stop offset="0.52" stop-color="#fff" stop-opacity="0.5"/>'
            '<stop offset="1" stop-color="#fff" stop-opacity="0"/>'
            f'</linearGradient><clipPath id="edge"><path d="{outline}"/>'
            '</clipPath></defs>'
            f'<path d="{outline}" fill="url(#tissue)" '
            'stroke="#d994af" stroke-opacity="0.35" stroke-width="0.55"/>'
            '<g clip-path="url(#edge)">'
            '<path d="M25 54 C17 39 19 22 27 10 C22 29 26 43 29 54Z" '
            'fill="url(#fold)"/>'
            '<g fill="none" stroke="#d8a0b7" stroke-width="0.35" stroke-opacity="0.23">'
            '<path d="M26 52 C16 39 10 24 12 11 M26 52 C22 34 18 20 20 9 '
            'M27 52 C28 36 33 20 34 9 M27 51 C36 35 41 26 40 17"/>'
            '<path d="M18 37 Q10 31 6 25 M15 29 Q11 20 10 17 '
            'M23 33 Q27 22 27 16 M33 37 Q40 29 43 24" stroke-opacity="0.5"/>'
            '</g></g></svg>'
        )
        rules.append(
            f'.fp-cherry-blossoms i:nth-child(3n + {variant + 1}) b{{'
            f'background-image: url("data:image/svg+xml,{petal}");}}'
        )
    for i in range(count):
        size = random.uniform(24, 34)
        rules.append(
            f".fp-cherry-blossoms i:nth-child({i + 1}){{"
            f"left: {random.uniform(1, 95):.1f}%;"
            f"top: {random.uniform(-8, 72):.1f}vh;"
            f"width: {size:.1f}px; height: {size * 1.2:.1f}px;"
            f"--fp-wind: {random.uniform(25, 90):.1f}px;"
            f"--fp-fall: {random.uniform(135, 230):.1f}px;"
            f"--fp-tilt: {random.uniform(-160, 160):.1f}deg;"
            f"--fp-flutter: {random.uniform(-18, 18):.1f}px;"
            f"animation-duration: {random.uniform(2.6, 3.1):.2f}s;"
            f"animation-delay: {random.uniform(0, 0.4):.2f}s;"
            "}"
        )
    return (_style(rules) + f'<div class="{CELEBRATION_CLASS} fp-cherry-blossoms" aria-hidden="true">'
            + "<i><b></b></i>" * count + "</div>")


def _rocket_launch_html(puffs=26):
    """An illustrated rocket lifting off along a curved trail of fading exhaust."""
    rocket = quote(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 140">'
        '<defs><linearGradient id="hull">'
        '<stop stop-color="#c3d3e4"/><stop offset="0.35" stop-color="#fff"/>'
        '<stop offset="0.7" stop-color="#e9f0f7"/><stop offset="1" stop-color="#94acc7"/>'
        '</linearGradient><linearGradient id="glass" x2="1" y2="1">'
        '<stop stop-color="#b4f4ff"/><stop offset="1" stop-color="#2084c8"/>'
        '</linearGradient></defs>'
        '<path d="M23 77 L8 102 L8 130 L29 111Z" fill="#ec5962" stroke="#b93649"/>'
        '<path d="M57 77 L72 102 L72 130 L51 111Z" fill="#d84958" stroke="#b93649"/>'
        '<path d="M40 4 C24 21 17 55 22 100 L28 118 H52 L58 100 '
        'C63 55 56 21 40 4Z" fill="url(#hull)" stroke="#66819d" stroke-width="1.5"/>'
        '<path d="M40 4 C32 13 26 26 23 39 Q40 45 57 39 '
        'C54 26 48 13 40 4Z" fill="#f2676b"/>'
        '<path d="M40 8 Q48 24 50 41 L57 39 Q51 18 40 8Z" fill="#d94b5c"/>'
        '<circle cx="40" cy="66" r="14" fill="#405e7c"/>'
        '<circle cx="40" cy="66" r="10.5" fill="url(#glass)"/>'
        '<path d="M33 65 Q33 59 40 58" fill="none" stroke="#fff" '
        'stroke-width="2.5" stroke-linecap="round" opacity="0.8"/>'
        '<path d="M29 101 H51" stroke="#a4b8cc" stroke-width="1.5"/>'
        '<rect x="29" y="118" width="22" height="10" rx="3" fill="#40546e"/>'
        '<path d="M40 94 L36 119 L40 133 L44 119Z" fill="#ed626a"/>'
        '</svg>'
    )
    launch_x = random.uniform(34, 54)
    # Shared flight positions keep each exhaust puff on the rocket's trajectory.
    path = []
    for percent in range(0, 101, 10):
        travel = max(0, percent / 100 - 0.12)
        path.append((percent, 26 * travel ** 3, 220 * travel ** 2, 24 * travel))
    rules = [
        _SHARED_RULES,
        """.fp-rocket-craft {
            --fp-rocket-size: clamp(60px, 7vw, 80px);
            position: absolute; bottom: 8vh; width: var(--fp-rocket-size);
            margin-left: calc(var(--fp-rocket-size) * -0.5); aspect-ratio: 4 / 7;
            transform-origin: 50% 90%; opacity: 0;
            animation-name: fp-rocket-flight;
            animation-duration: 4.00s;
            animation-timing-function: linear;
            animation-fill-mode: forwards;
        }
        .fp-rocket-body {
            position: absolute; inset: 0; z-index: 1;
            background-size: 100% 100%; background-repeat: no-repeat;
            filter: drop-shadow(0 2px 3px rgba(36, 63, 94, 0.25));
        }
        .fp-rocket-flame {
            position: absolute; left: 35%; top: 87%; width: 30%; height: 42%;
            border-radius: 45% 45% 50% 50% / 12% 12% 90% 90%;
            background: linear-gradient(to bottom, #fffbe9, #ffd15b 35%, #ff8a45 65%, transparent);
            filter: drop-shadow(0 0 5px rgba(255, 172, 66, 0.7));
            transform-origin: top; opacity: 0;
            animation-name: fp-rocket-ignition;
            animation-duration: inherit;
            animation-timing-function: ease-in-out;
            animation-fill-mode: forwards;
        }
        .fp-rocket-flame::before {
            content: ""; position: absolute; left: 30%; top: 0; width: 40%; height: 66%;
            border-radius: 0 0 50% 50%; background: #fffdf0;
        }
        .fp-rocket-launch > i {
            position: absolute; display: block; border-radius: 50%; opacity: 0;
            background: radial-gradient(circle at 35% 30%, rgba(246, 249, 255, 0.85),
                rgba(167, 190, 215, 0.5) 50%, rgba(156, 181, 209, 0.1) 70%, transparent 74%);
            animation-name: fp-rocket-exhaust;
            animation-timing-function: ease-out;
            animation-fill-mode: forwards;
        }
        @keyframes fp-rocket-ignition {
            0% { transform: scaleY(0.1); opacity: 0; }
            10% { transform: scaleY(0.35); opacity: 0.8; }
            20% { transform: scaleY(0.85); opacity: 1; }
            30%, 50%, 70% { transform: scaleY(1.1); opacity: 1; }
            40%, 60%, 80% { transform: scaleY(0.85); opacity: 1; }
            100% { transform: scaleY(0.7); opacity: 0; }
        }
        @keyframes fp-rocket-exhaust {
            0% { transform: translate(-50%, 50%) scale(0.25); opacity: 0; }
            18% { opacity: 0.8; }
            100% { transform: translate(calc(-50% + var(--fp-smoke-drift)),
                calc(50% + 36px)) scale(1.8); opacity: 0; }
        }""",
        f'.fp-rocket-body{{background-image: url("data:image/svg+xml,{rocket}");}}',
        f".fp-rocket-craft{{left: {launch_x:.2f}%;}}",
        "@keyframes fp-rocket-flight {" + "".join(
            f"{percent}%{{transform: translate({x:.2f}vw, {-y:.2f}vh) rotate({tilt:.1f}deg);"
            f"opacity: {0 if percent in (0, 100) else 1};}}"
            for percent, x, y, tilt in path
        ) + "}",
    ]
    for i in range(puffs):
        delay = 0.3 + 2.4 * i / max(puffs - 1, 1)
        phase = delay / 4 * 100
        for start, end in zip(path, path[1:]):
            if start[0] <= phase <= end[0]:
                fraction = (phase - start[0]) / (end[0] - start[0])
                x = start[1] + fraction * (end[1] - start[1])
                y = start[2] + fraction * (end[2] - start[2])
                break
        size = random.uniform(16, 30)
        rules.append(
            f".fp-rocket-launch > i:nth-child({i + 1}){{"
            f"left: {launch_x + x:.2f}%; bottom: {8 + y:.2f}vh;"
            f"width: {size:.1f}px; height: {size:.1f}px;"
            f"--fp-smoke-drift: {random.uniform(-28, 28):.1f}px;"
            f"animation-duration: {random.uniform(0.9, 1.3):.2f}s;"
            f"animation-delay: {delay:.2f}s;"
            "}"
        )
    return (_style(rules) + f'<div class="{CELEBRATION_CLASS} fp-rocket-launch" aria-hidden="true">'
            + "<i></i>" * puffs
            + '<span class="fp-rocket-craft"><b class="fp-rocket-body"></b>'
              '<b class="fp-rocket-flame"></b></span></div>')


def _landing_plane_html():
    """A passenger jet makes a single horizontal landing with tire smoke."""
    # Public-domain aircraft and adapted landing motion; credits accompany the assets.
    plane = "data:image/svg+xml," + quote(_celebration_asset("landing-plane.svg"))
    css = _celebration_asset("landing-plane.css").replace("__PLANE_SVG_URI__", plane)
    return _style([_SHARED_RULES, css]) + _celebration_asset("landing-plane.html")


def _arriving_train_html():
    """The bundled CSS steam train crosses to a station at the right edge."""
    station = "data:image/svg+xml," + quote(_celebration_asset("train-station.svg"))
    css = _celebration_asset("arriving-train.css").replace("__STATION_SVG_URI__", station)
    return _style([_SHARED_RULES, css]) + _celebration_asset("arriving-train.html")


def _emoji_rain_html(count=100):
    """Happy emoji drifting down together in a 4.5-second shower."""
    picks = []
    while len(picks) < count:
        picks.extend(random.sample(_RAIN_EMOJIS, min(count - len(picks), len(_RAIN_EMOJIS))))
    rules = [
        _SHARED_RULES,
        """.fp-emoji-rain i {
            position: absolute; top: -14vh; font-style: normal; line-height: 1;
            /* Keep a steady fall as later emoji join the same shower. */
            animation-timing-function: linear;
            animation-iteration-count: 1;
            animation-fill-mode: forwards;
        }
        @keyframes fp-emoji-a {
            0% { transform: translateX(0) rotate(-16deg); opacity: 1; }
            80% { opacity: 1; }
            100% { transform: translateX(4vw) translateY(118vh) rotate(20deg); opacity: 0; }
        }
        @keyframes fp-emoji-b {
            0% { transform: translateX(0) rotate(14deg); opacity: 1; }
            80% { opacity: 1; }
            100% { transform: translateX(-5vw) translateY(118vh) rotate(-24deg); opacity: 0; }
        }""",
    ]
    for i, _ in enumerate(picks):
        # Varied fall times preserve a scattered shower rather than a single row.
        # One full-length fall anchors the cue at 4.5 seconds; the rest finish by it.
        duration = 4.5 if i == 0 else round(random.uniform(3.5, 4.2), 2)
        delay = round(random.uniform(0, 4.5 - duration), 2)
        rules.append(
            f".fp-emoji-rain i:nth-child({i + 1}){{"
            f"left: {random.uniform(1, 96):.2f}%;"
            f"font-size: {random.uniform(1.1, 2.3):.2f}rem;"
            f"animation-name: fp-emoji-{'ab'[i % 2]};"
            f"animation-duration: {duration:.2f}s;"
            f"animation-delay: {delay:.2f}s;"
            "}"
        )
    body = "".join(f"<i>{emoji}</i>" for emoji in picks)
    return _style(rules) + f'<div class="{CELEBRATION_CLASS} fp-emoji-rain">' + body + "</div>"


CSS_EFFECT_BUILDERS = {
    CONFETTI: _confetti_html,
    FIREWORKS: _fireworks_html,
    BUBBLES: _bubbles_html,
    EMOJI_RAIN: _emoji_rain_html,
    SHOOTING_STARS: _shooting_stars_html,
    CHERRY_BLOSSOMS: _cherry_blossoms_html,
    ROCKET_LAUNCH: _rocket_launch_html,
    LANDING_PLANE: _landing_plane_html,
    ARRIVING_TRAIN: _arriving_train_html,
}

EFFECT_NAMES = (BALLOONS, *CSS_EFFECT_BUILDERS)


def choose_effect(previous=None, names=EFFECT_NAMES):
    """Pick an effect at random, skipping the one played last so none repeats back to back."""
    options = [name for name in names if name != previous] or list(names)
    return random.choice(options)


def render_effect(name):
    """Play one named effect."""
    if name == BALLOONS:
        # Keep Streamlit's balloon artwork, extending the native 750ms flight.
        # Its stagger is 0–1s; the final balloon anchors the total at four seconds.
        st.html(_style(["""
            [data-testid="stBalloons"] img {
                animation-duration: 3s; animation-fill-mode: both;
            }
            [data-testid="stBalloons"] img:last-child { animation-delay: 1s; }
            @media (prefers-reduced-motion: reduce) {
                [data-testid="stBalloons"] { display: none; }
            }
        """]))
        st.balloons()
        return
    builder = CSS_EFFECT_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"unknown celebration effect: {name!r}")
    st.html(builder())


def celebrate():
    """Play a random celebratory effect, avoiding an immediate repeat. Returns the effect name."""
    name = choose_effect(st.session_state.get(LAST_EFFECT_KEY))
    st.session_state[LAST_EFFECT_KEY] = name
    render_effect(name)
    return name
