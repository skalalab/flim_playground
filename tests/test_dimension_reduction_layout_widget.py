"""Native chart sizing preserves canonical DR geometry and user plot settings."""

import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from src import dataset_io, export_script
from src.vis import multivar
from src.widgets import analysis_config_widgets as acw
from src.widgets import plot_layout


def _figure():
    axes = {"xaxis": [0., .4], "yaxis": [0., 1.],
            "xaxis2": [.5, .7], "yaxis2": [.5, 1.]}
    annotations = [dict(x=.6, y=1.04, xref="paper", yref="paper"),
                   dict(x=.5, y=.8, xref="x2 domain", yref="y2 domain")]
    return go.Figure(
        go.Scatter(x=[1., 3.], y=[2., 4.], marker=dict(size=14, opacity=.7)),
        layout=dict(
            xaxis=dict(domain=axes["xaxis"], range=[-2., 7.]),
            yaxis=dict(domain=axes["yaxis"], range=[-1., 8.]),
            xaxis2=dict(domain=axes["xaxis2"], matches="x"),
            yaxis2=dict(domain=axes["yaxis2"], matches="y"),
            annotations=[dict(**a, text="A", showarrow=False) for a in annotations],
            legend=dict(font=dict(size=18)),
            meta=dict(dimension_reduction_layout=dict(
                plot_height=.75, axes=axes, annotations=annotations)),
        ),
    )


def _widget_app(spec):
    import plotly.graph_objects as go
    from src.widgets.plot_layout import dimension_reduction_chart

    dimension_reduction_chart(go.Figure(spec), key="test_dimension_reduction")


@pytest.mark.parametrize("metadata", [True, False])
def test_wrapper_keeps_native_chart_metadata_and_styling(metadata):
    fig = _figure()
    if not metadata:
        fig.layout.meta = None
    before = fig.to_plotly_json()
    at = AppTest.from_function(_widget_app, args=(before,)).run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert spec == before
    assert len(at.get("plotly_chart")) == 1
    # Only figures carrying canonical DR geometry need the sizing script.
    assert bool(at.get("html")) is metadata


_BROWSER_HARNESS = r"""
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
let rootPresent = true;
let expanded = false;
let nextFrame = 0;
const frames = new Map();
const relayouts = [];
const resizeObservers = [];
const mutationObservers = [];
const container = {style: {}};
const fullscreen = {};
let squareCleanupCalls = 0;
let graph;
let annotationBoxes = input.annotationBoxes || [];
let availableWidth = input.availableWidth || input.layout._size.w + input.layout._size.l + input.layout._size.r;
const windowListeners = new Map();
const root = {querySelector: () => graph, getBoundingClientRect: () => ({width: availableWidth})};
global.document = {body: {}, querySelector: () => rootPresent ? root : null};
global.getComputedStyle = () => ({position: expanded ? 'fixed' : 'relative'});
global.requestAnimationFrame = fn => {const id = ++nextFrame; frames.set(id, fn); return id;};
global.cancelAnimationFrame = id => frames.delete(id);
global.ResizeObserver = class {
    constructor(fn) {this.fn = fn; this.targets = new Set(); resizeObservers.push(this);}
    observe(target) {this.targets.add(target);}
    disconnect() {this.targets.clear(); this.disconnected = true;}
};
global.MutationObserver = class {
    constructor(fn) {this.fn = fn; mutationObservers.push(this);}
    observe() {}
    disconnect() {this.disconnected = true;}
};
function makeGraph() {
    return {
        _fullLayout: JSON.parse(JSON.stringify(input.layout)),
        events: new Map(),
        on(name, fn) {this.events.set(name, fn);},
        removeListener(name, fn) {if (this.events.get(name) === fn) this.events.delete(name);},
        closest(selector) {return selector.includes('stFullScreenFrame') ? fullscreen : container;},
        getBoundingClientRect() {return {left: 125};},
        querySelectorAll() {
            const current = this;
            return annotationBoxes.map(box => ({getBoundingClientRect() {
                const layout = current._fullLayout;
                const annotation = layout.annotations[box.index];
                const left = 125 + layout._size.l + layout._size.w * annotation.x
                    + (annotation.xshift || 0) - box.width * (annotation.xanchor === 'center' ? .5 : 0);
                return {left, right: left + box.width, width: box.width};
            }}));
        },
    };
}
graph = makeGraph();
global.window = {
    innerHeight: input.viewportHeight || 2000,
    addEventListener(name, fn) {
        if (!windowListeners.has(name)) windowListeners.set(name, new Set());
        windowListeners.get(name).add(fn);
    },
    removeEventListener(name, fn) {windowListeners.get(name)?.delete(fn);},
    _flimSquare2dCleanup: () => squareCleanupCalls++,
    Plotly: {relayout(target, changes) {
        relayouts.push(JSON.parse(JSON.stringify(changes)));
        for (const [key, value] of Object.entries(changes)) {
            const annotation = key.match(/^annotations\[(\d+)\]\.(x|y)$/);
            if (annotation) target._fullLayout.annotations[+annotation[1]][annotation[2]] = value;
            else {
                const [axis, property] = key.split('.');
                target._fullLayout[axis][property] = value;
                if (key === 'margin.r') {
                    const size = target._fullLayout._size;
                    size.w += size.r - value;
                    size.r = value;
                }
            }
        }
        target.events.get('plotly_afterplot')?.();
        return Promise.resolve();
    }},
};
function flush() {
    let count = 0;
    while (frames.size) {
        if (++count > 20) throw Error('animation/relayout loop');
        const [id, fn] = frames.entries().next().value;
        frames.delete(id); fn();
    }
}
function resize(width, height) {
    Object.assign(graph._fullLayout._size, {w: width, h: height});
    if (!expanded) availableWidth = width + graph._fullLayout._size.l + graph._fullLayout._size.r;
    for (const observer of resizeObservers) if (observer.targets.size) observer.fn();
    flush();
}
function snapshot() {
    return JSON.parse(JSON.stringify({height: container.style.height, width: container.style.width,
                                     layout: graph._fullLayout,
                                     relayouts, resizeTargets: resizeObservers.at(-1)?.targets.size}));
}
function viewportResize(height, width = availableWidth) {
    // Native Plotly has resized to the preceding container dimensions before
    // the user changes the viewport again.
    const size = graph._fullLayout._size;
    size.w = parseFloat(container.style.width) - size.l - size.r;
    size.h = parseFloat(container.style.height) - size.t - size.b;
    window.innerHeight = height;
    availableWidth = width;
    for (const fn of windowListeners.get('resize') || []) fn();
    flush();
}
eval(input.script);
flush();
const initial = snapshot();
let result;
if (input.scenario === 'normal_resize') {
    resize(1000, 650);
    result = {initial, resized: snapshot(), squareCleanupCalls};
} else if (input.scenario === 'fullscreen') {
    expanded = true; resize(1200, 450);
    const wide = snapshot();
    resize(600, 1000);
    const tall = snapshot();
    expanded = false; resize(800, 600);
    result = {initial, wide, tall, restored: snapshot()};
} else if (input.scenario === 'replacement_cleanup') {
    const old = graph;
    graph = makeGraph();
    mutationObservers.at(-1).fn(); flush();
    const rebound = {oldListener: old.events.has('plotly_afterplot'),
                     newListener: graph.events.has('plotly_afterplot')};
    rootPresent = false;
    mutationObservers.at(-1).fn(); flush();
    result = {rebound, listener: graph.events.has('plotly_afterplot'),
              resizeDisconnected: resizeObservers.every(o => o.disconnected),
              mutationDisconnected: mutationObservers.every(o => o.disconnected),
              cleanupPresent: !!window._flimDimensionReductionCleanup, pendingFrames: frames.size,
              resizeListeners: windowListeners.get('resize')?.size || 0};
} else if (input.scenario === 'duplicate_script') {
    eval(input.script); flush();
    result = {activeResizeObservers: resizeObservers.filter(o => o.targets.size).length,
              disconnectedMutations: mutationObservers.filter(o => o.disconnected).length,
              listener: graph.events.has('plotly_afterplot'), squareCleanupCalls};
} else if (input.scenario === 'right_margin') {
    annotationBoxes = annotationBoxes.map(box => ({...box, width: box.grownWidth || box.width}));
    graph.events.get('plotly_afterplot')?.(); flush();
    result = {initial, grown: snapshot()};
} else if (input.scenario === 'viewport_resize') {
    viewportResize(1200);
    const taller = snapshot();
    viewportResize(2000, 1600);
    const wider = snapshot();
    viewportResize(500, 1600);
    result = {initial, taller, wider, shorter: snapshot()};
}
process.stdout.write(JSON.stringify(result));
"""


def _browser(scenario, *, layout=None, annotation_boxes=None, viewport_height=2000, available_width=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the chart sizing JavaScript")
    source = inspect.getsource(plot_layout.dimension_reduction_chart)
    script = re.search(r"<script>(.*?)</script>", source, re.S).group(1)
    if layout is None:
        layout = _figure().to_plotly_json()["layout"]
        layout["_size"] = dict(w=800, h=360, l=60, r=20, t=36, b=44)
    result = subprocess.run([node, "-e", _BROWSER_HARNESS],
                            input=json.dumps(dict(script=script, layout=layout, scenario=scenario,
                                                  annotationBoxes=annotation_boxes,
                                                  viewportHeight=viewport_height,
                                                  availableWidth=available_width)),
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_normal_height_uses_rendered_inner_width_and_margins():
    result = _browser("normal_resize")
    assert result["initial"]["height"] == "680px"
    assert result["resized"]["height"] == "830px"
    assert result["initial"]["resizeTargets"] >= 2
    assert not result["resized"]["relayouts"]
    assert result["squareCleanupCalls"] == 0


def test_fullscreen_fits_domains_and_paper_annotations_then_restores_canonical_values():
    result = _browser("fullscreen")
    wide = result["wide"]["layout"]
    tall = result["tall"]["layout"]
    base = result["initial"]["layout"]
    assert result["wide"]["height"] == result["initial"]["height"]
    assert result["tall"]["height"] == result["initial"]["height"]
    assert wide["xaxis"]["domain"] == pytest.approx([.25, .45])
    assert wide["yaxis"]["domain"] == [0., 1.]
    assert wide["annotations"][0]["x"] == pytest.approx(.55)
    assert tall["xaxis"]["domain"] == [0., .4]
    assert tall["yaxis"]["domain"] == pytest.approx([.275, .725])
    assert tall["annotations"][0]["y"] == pytest.approx(.743)
    for state in [wide, tall, result["restored"]["layout"]]:
        assert state["xaxis"]["range"] == base["xaxis"]["range"]
        assert state["xaxis2"]["matches"] == "x"
        assert state["annotations"][1] == base["annotations"][1]
        assert state["legend"] == base["legend"]
        assert state["meta"] == base["meta"]
    for axis in base["meta"]["dimension_reduction_layout"]["axes"]:
        assert result["restored"]["layout"][axis]["domain"] == base[axis]["domain"]
    assert result["restored"]["layout"]["annotations"] == base["annotations"]
    assert len(result["restored"]["relayouts"]) == 3


def test_graph_replacement_rebinds_listeners_and_removal_cleans_up_observers():
    result = _browser("replacement_cleanup")
    assert result["rebound"] == dict(oldListener=False, newListener=True)
    assert not result["listener"]
    assert result["resizeDisconnected"] and result["mutationDisconnected"]
    assert not result["cleanupPresent"] and not result["pendingFrames"]
    assert result["resizeListeners"] == 0


def test_fragment_reruns_replace_the_previous_observers():
    result = _browser("duplicate_script")
    assert result == dict(activeResizeObservers=1, disconnectedMutations=1,
                          listener=True, squareCleanupCalls=0)


def _margin_layout():
    layout = _figure().to_plotly_json()["layout"]
    layout["xaxis2"]["domain"] = [.6, 1.]
    layout["annotations"] = [
        dict(x=1., y=.5, xref="paper", yref="paper", xanchor="left", xshift=6),
        dict(x=.8, y=1., xref="paper", yref="paper", xanchor="center"),
    ]
    canonical = layout["meta"]["dimension_reduction_layout"]
    canonical["axes"]["xaxis2"] = [.6, 1.]
    canonical["annotations"] = layout["annotations"]
    layout["margin"] = dict(l=60, r=140, t=36, b=90, autoexpand=True)
    layout["_size"] = dict(w=800, h=360, l=60, r=140, t=36, b=90)
    return layout


def test_right_margin_follows_row_label_width_and_font_changes():
    result = _browser("right_margin", layout=_margin_layout(), annotation_boxes=[
        dict(index=0, width=20, grownWidth=120), dict(index=1, width=40)])
    assert result["initial"]["layout"]["margin"]["r"] == 38
    assert result["grown"]["layout"]["margin"]["r"] == 138
    for stage in [result["initial"], result["grown"]]:
        assert stage["layout"]["margin"]["b"] == 90
        assert stage["layout"]["margin"]["autoexpand"] is True
        assert stage["layout"]["xaxis"]["range"] == [-2., 7.]


def test_wide_column_heading_reserves_its_overhang_and_converges():
    result = _browser("right_margin", layout=_margin_layout(), annotation_boxes=[
        dict(index=0, width=20), dict(index=1, width=500)])
    margin = result["initial"]["layout"]["margin"]["r"]
    assert 92 <= margin <= 94
    assert len(result["initial"]["relayouts"]) <= 5
    assert result["grown"]["relayouts"] == result["initial"]["relayouts"]


def test_fullscreen_aspect_gutter_does_not_erase_row_label_allowance():
    result = _browser("fullscreen", layout=_margin_layout(), annotation_boxes=[
        dict(index=0, width=20), dict(index=1, width=40)])
    for stage in ["initial", "wide", "tall", "restored"]:
        assert result[stage]["layout"]["margin"]["r"] == 38
    assert result["wide"]["height"] == result["initial"]["height"]
    assert result["restored"]["layout"]["xaxis2"]["domain"] == [.6, 1.]


def test_overview_only_retains_space_for_its_right_hand_legend():
    layout = _margin_layout()
    canonical = layout["meta"]["dimension_reduction_layout"]
    canonical["axes"] = {key: value for key, value in canonical["axes"].items()
                         if key in ("xaxis", "yaxis")}
    result = _browser("right_margin", layout=layout, annotation_boxes=[dict(index=0, width=20)])
    assert result["initial"]["layout"]["margin"]["r"] == 140
    assert not result["initial"]["relayouts"]


def test_normal_chart_caps_height_and_regrows_from_available_parent_width():
    result = _browser("viewport_resize", viewport_height=700, available_width=1400)
    assert result["initial"]["height"] == "595px"
    assert result["initial"]["width"] == "767px"
    # A height-only viewport change still schedules a new size, even though the
    # root width is unchanged and the previous chart was much narrower.
    assert result["taller"]["height"] == "1020px"
    assert result["taller"]["width"] == "1333px"
    assert result["wider"]["height"] == "1220px"
    assert result["wider"]["width"] == "1600px"
    assert result["shorter"]["height"] == "425px"
    assert result["shorter"]["width"] == "540px"
    assert not result["shorter"]["relayouts"]


def test_large_legend_and_title_margins_leave_a_useful_minimum_plot_area():
    layout = _margin_layout()
    layout["_size"].update(t=160, b=400)
    result = _browser("right_margin", layout=layout, viewport_height=700, available_width=1000)
    assert result["initial"]["height"] == "680px"
    assert result["initial"]["width"] == "360px"
    # A genuinely narrow parent still bounds that minimum, without adding
    # overflow width or empty space inside the chart.
    narrow = _browser("right_margin", layout=layout, viewport_height=700, available_width=300)
    assert narrow["initial"]["height"] == "635px"
    assert narrow["initial"]["width"] == "300px"


def test_fullscreen_does_not_apply_the_normal_viewport_height_cap():
    result = _browser("fullscreen", viewport_height=500, available_width=1400)
    for stage in ["wide", "tall"]:
        assert result[stage]["height"] == result["initial"]["height"]
        assert result[stage]["width"] == result["initial"]["width"]
    assert result["wide"]["layout"]["xaxis"]["domain"] == pytest.approx([.25, .45])
    assert result["tall"]["layout"]["yaxis"]["domain"] == pytest.approx([.275, .725])


def test_data_analysis_routes_dimension_reduction_through_responsive_wrapper(monkeypatch):
    frame = pd.DataFrame(dict(cell_id=["a", "b", "c"], first=[1., 3., 2.], second=[3., 2., 1.]))
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: [])
    monkeypatch.setattr(acw, "get_fov_name_col_analysis", lambda *a, **k: None)
    monkeypatch.setattr(acw, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        frame.copy(), {"Uncategorized Features": ["first", "second"]}, True, ",", "cell_id"))
    monkeypatch.setattr(multivar, "dimension_reduction_plot", lambda *a, **k: _figure())
    monkeypatch.setattr(export_script, "generate_script", lambda state: "# test")
    seen = []
    monkeypatch.setattr(plot_layout, "dimension_reduction_chart",
                        lambda fig, *, key: seen.append((fig, key)))
    page = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
    at = AppTest.from_file(page).run(timeout=30)
    at.radio[0].set_value("### **Multivariate**")
    at.session_state["analysis_control_dr_method"] = "PCA"
    at.session_state["ms_Uncategorized Features"] = ["first", "second"]
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    assert len(seen) == 1
    assert seen[0][1] == "plot_chart_Dimension Reduction"
    assert seen[0][0].layout.meta == _figure().layout.meta
