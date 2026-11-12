"""Check when the plot fragment escalates to a full rerun.

A colormap, group-count or axis-label change makes the base figure stale, so the
fragment reruns the app before copying or drawing it: drawing the stale figure
first cost about 0.7 s on the server plus a browser redraw at 200k points.
Requests raised by widgets inside the fragment (the reorder flag, a facet click)
can only be read after those widgets render, so that escalation stays last.

An early st.rerun() is safe from Streamlit 1.64, which skips stale-widget cleanup
when a run stops for a rerun; before that, widgets the fragment had not rendered
yet lost their state. pyproject.toml must therefore require 1.64 or newer.
These source checks complement the page's AppTest coverage, which cannot run a
fragment on its own.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILS = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)

page = (ROOT / "pages" / "data_analysis.py").read_text(encoding="utf-8")
tree = ast.parse(page)

fragment = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_render_plot_and_controls":
        fragment = node
check("the plot fragment exists", fragment is not None)


def call_name(node):
    func = node.func
    return getattr(func, "attr", None) or getattr(func, "id", None)


def mentions_build_params(node):
    names = {getattr(n, "id", None) for n in ast.walk(node)}
    return {"_plot_build_params", "build_params"} <= names


if fragment:
    reruns, widgets, draws = [], [], []
    for node in ast.walk(fragment):
        if isinstance(node, ast.Call):
            name = call_name(node)
            if name == "rerun":
                reruns.append(node)
            if name in ("plot_config_widget", "reorder_x_axis_widget"):
                widgets.append((name, node.lineno))
            # The stale-figure work: copying the base figure and drawing any chart.
            if (name == "Figure" or name in ("dimension_reduction_chart", "square_2d_plot",
                                             "phasor_chart", "plotly_chart")):
                draws.append(node.lineno)
    check("the fragment renders the styling widget", any(n == "plot_config_widget" for n, _ in widgets), widgets)
    check("the fragment escalates in two places", len(reruns) == 2, [r.lineno for r in reruns])

    early = [r for r in reruns
             if any(isinstance(parent, ast.If) and mentions_build_params(parent.test)
                    and r in ast.walk(parent) for parent in ast.walk(fragment))]
    first_draw = min(draws, default=0)
    check("a stale base figure reruns the app before it is copied or drawn",
          len(early) == 1 and early[0].lineno < first_draw
          and any(k.arg == "scope" and getattr(k.value, "value", None) == "app"
                  for k in early[0].keywords),
          f"early reruns {[r.lineno for r in early]}, first draw at {first_draw}")

    late = [r for r in reruns if r not in early]
    last_widget = max((line for _n, line in widgets), default=0)
    check("in-fragment requests escalate after every widget in the fragment",
          len(late) == 1 and late[0].lineno > last_widget,
          f"late reruns {[r.lineno for r in late]}, last widget at {last_widget}")

widgets_src = (ROOT / "src" / "widgets" / "visualization_widgets.py").read_text(encoding="utf-8")
wtree = ast.parse(widgets_src)
for node in ast.walk(wtree):
    if isinstance(node, ast.FunctionDef) and node.name == "reorder_x_axis_widget":
        inner = [n.lineno for n in ast.walk(node)
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "rerun"]
        check("the reorder widget never reruns on its own (it runs inside the fragment)",
              not inner, f"st.rerun() at {inner}")

check("a rebuild flag carries the reorder request to the late escalation",
      "_plot_needs_rebuild" in widgets_src and "_plot_needs_rebuild" in page)

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
floor = re.search(r'"streamlit>=(\d+)\.(\d+)', pyproject)
check("pyproject requires Streamlit 1.64+, which keeps unrendered widgets through st.rerun()",
      bool(floor) and (int(floor.group(1)), int(floor.group(2))) >= (1, 64),
      floor.group(0) if floor else "no streamlit>= floor found")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILING: {FAILS}"))
sys.exit(1 if FAILS else 0)
