"""Visual-encoding layout, evaluation order, labels, and state transitions. A recording
Streamlit stub checks container placement and keyed selections. Run with: python
tests/check_encoding_row.py.
"""
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.widgets.visualization_widgets as vw  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '   ' + str(detail)}")
    if not cond:
        FAILS.append(name)


class Slot:
    """A stubbed column or container. Records what is written into it, and when."""

    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def __enter__(self):
        self.rec.stack.append(self.name)
        return self

    def __exit__(self, *exc):
        self.rec.stack.pop()
        return False


class FakeStreamlit:
    def __init__(self, state):
        self.n_cols = None
        self.session_state = state
        self.events = []            # (kind, label, slot) in EVALUATION order
        self.slots = []             # slot names in CREATION (display) order
        self.slot_kw = {}           # slot name -> the kwargs it was created with
        self.slot_parent = {}       # slot name -> the slot it was created inside
        self.widget_kw = {}         # widget label -> the kwargs it was called with
        self.widget_options = {}    # widget label -> the options it was OFFERED
        self.visible = []           # labels the USER can read, in display order
        self.rendered_keys = set()  # keys a widget actually rendered under this run
        self.stack = []

    def _slot(self, name, **kw):
        # Placement checks need container parentage and layout kwargs.
        self.slots.append(name)
        self.slot_kw[name] = kw
        self.slot_parent[name] = self.stack[-1] if self.stack else None
        return Slot(self, name)

    def columns(self, spec, **kw):
        # Streamlit accepts either a count or per-column weights.
        self.n_cols = spec if isinstance(spec, int) else len(spec)
        self.column_weights = [1] * spec if isinstance(spec, int) else list(spec)
        self.column_kw = kw
        return [self._slot(f"col{i}") for i in range(self.n_cols)]

    def container(self, **kw):
        return self._slot(f"container{len(self.slots)}", **kw)

    def markdown(self, body, **kw):
        """Record visible text, excluding HTML tags and CSS."""
        text = re.sub(r"<style\b[^>]*>.*?</style>", "", body, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", "", text).strip()
        self.events.append(("markdown", text, self.stack[-1] if self.stack else None))
        self.visible.append(text)

    def html(self, body, **kw):
        self.events.append(("html", body, self.stack[-1] if self.stack else None))

    def _widget(self, kind, label, key, default, label_visibility="visible"):
        """Keyed widgets read and update session state; unkeyed widgets return their
        default.
        """
        where = self.stack[-1] if self.stack else None
        self.events.append((kind, label, where))
        if key is not None:
            self.rendered_keys.add(key)
        # Collapsed labels do not count toward visible label uniqueness.
        if label_visibility != "collapsed":
            self.visible.append(label)
        if key is not None:
            value = self.session_state.get(key, default)
            self.session_state[key] = value
            return value
        return default

    def selectbox(self, label, options, index=None, key=None, disabled=False,
                  label_visibility="visible", **kw):
        self.widget_kw[label] = {**kw, "disabled": disabled, "label_visibility": label_visibility}
        self.widget_options[label] = list(options)
        # Disabled widgets still return their retained selection.
        value = self._widget("selectbox", label, key, None, label_visibility)
        return value if value in options else None

    def multiselect(self, label, options, default=None, key=None, **kw):
        self.widget_kw[label] = kw
        self.widget_options[label] = list(options)
        value = self._widget("multiselect", label, key, list(default or []),
                             kw.get("label_visibility", "visible"))
        return [v for v in (value or []) if v in options]

    def toggle(self, label, key=None, **kw):
        self.widget_kw[label] = kw
        return bool(self._widget("toggle", label, key, False,
                                 kw.get("label_visibility", "visible")))

    def segmented_control(self, label, options, default=None, key=None, **kw):
        self.widget_kw[label] = kw
        self.widget_options[label] = list(options)
        return self._widget("segmented_control", label, key, default,
                            kw.get("label_visibility", "visible"))

    def purge_unrendered(self, keys):
        """Simulate missing widget state by key, including the picker shared by shape and
        subcolor.
        """
        for key in keys:
            if key not in self.rendered_keys:
                # None also catches restores that incorrectly rely on dict.get's fallback.
                self.session_state[key] = None


DF = pd.DataFrame({
    "experiment": ["E1", "E2"] * 4,
    "patient_id": ["P1", "P2", "P3", "P4"] * 2,
    "treatment": ["T1", "T2"] * 4,
    "dish": ["d1", "d2"] * 4,
})
CATS = ["experiment", "patient_id", "treatment", "dish"]


def run(state=None, separate=True, match=True, collapse=False):
    fake = FakeStreamlit(dict(state or {}))
    real = vw.st
    vw.st = fake
    try:
        result = vw.visual_encoding_channels_widget(
            DF, CATS, color_based=True, point_based=True,
            separate_by_available=separate, subcolor_available=match,
            collapse_available=collapse,
        )
    finally:
        vw.st = real
    return fake, result


def label_of(fake, kind):
    return next((lab for k, lab, _ in fake.events if k == kind), None)


def order_of(fake, kind, label=None):
    for i, (k, lab, _) in enumerate(fake.events):
        if k == kind and (label is None or lab == label):
            return i
    return -1


def column_of(fake, kind, label):
    """Find a widget's display column through keyed containers."""
    slot = next(where for k, lab, where in fake.events if k == kind and lab == label)
    while slot is not None and not slot.startswith("col"):
        slot = fake.slot_parent.get(slot)
    return slot


print("1. Feature Comparison has four aligned columns")
fake, _ = run(collapse=True)
check("four columns, with extra width for the selector",
      fake.column_weights == [1, 1, 1, 1.4], fake.column_weights)
check("pickers align along their bottom edge",
      fake.column_kw.get("vertical_alignment") == "bottom", fake.column_kw)
check("Collapse by is third", column_of(fake, "selectbox", "Collapse by") == "col2")
check("selector and shared picker occupy the fourth column",
      column_of(fake, "segmented_control", "Point encoding") == "col3"
      and column_of(fake, "selectbox", "Shape by") == "col3")
check("no standalone opacity in Feature Comparison",
      "Opacity by" not in fake.widget_options, fake.widget_options)
check("the row has a stable CSS scope",
      any(kw.get("key") == "vis_encoding_fc_row" for kw in fake.slot_kw.values()))
check("full mode names are available from one native control",
      fake.widget_options["Point encoding"] == ["opacity", "subcolor", "shape"]
      and [fake.widget_kw["Point encoding"]["format_func"](m)
           for m in fake.widget_options["Point encoding"]] == ["Opacity", "Subcolor", "Shape"])
check("native accessible labels are retained without taking another row",
      fake.widget_kw["Point encoding"].get("label_visibility") == "collapsed"
      and fake.widget_kw["Shape by"].get("label_visibility") == "collapsed")
fake, _ = run(match=False)
check("other methods retain separate opacity and shape controls",
      fake.n_cols == 4 and "Opacity by" in fake.widget_options
      and "Shape by" in fake.widget_options and "Point encoding" not in fake.widget_options)

print("2. mode selects one channel and relabels grouping in the same run")
for mode, active_index in (("shape", 2), ("subcolor", 4), ("opacity", 1)):
    label = f"{mode.title()} by"
    fake, result = run(state={vw.POINT_MODE_KEY: mode,
                              vw.COLOR_BY_KEY: ["experiment"],
                              vw.PICKER_COL_KEY: "patient_id",
                              vw.OPACITY_BY_KEY: "dish"}, collapse=True)
    check(f"{mode}: only the chosen role is returned",
          result[active_index] == "patient_id"
          and all(result[i] is None for i in (1, 2, 4) if i != active_index), result)
    check(f"{mode}: mode is evaluated before the colour label",
          order_of(fake, "segmented_control") < order_of(fake, "multiselect"), fake.events)
    check(f"{mode}: shared picker follows its mode selector",
          order_of(fake, "segmented_control") < order_of(fake, "selectbox", label), fake.events)
    check(f"{mode}: correct grouping label",
          label_of(fake, "multiselect") == ("Group by" if mode == "subcolor" else "Color by"))
    check(f"{mode}: only one decoration picker",
          set(fake.widget_options).intersection({"Shape by", "Subcolor by", "Opacity by"}) == {label})
    check(f"{mode}: independent opacity is retained for other methods",
          fake.session_state[vw.OPACITY_BY_KEY] == "dish")

fake, result = run(state={vw.POINT_MODE_KEY: "subcolor", vw.COLOR_BY_KEY: []})
check("Subcolor relabels even without a field or groups",
      label_of(fake, "multiselect") == "Group by")
check("Subcolor requires a group", fake.widget_kw["Subcolor by"]["disabled"] and result[4] is None)
fake, result = run(state={vw.POINT_MODE_KEY: "subcolor", vw.COLOR_BY_KEY: [],
                          vw.PICKER_COL_KEY: "patient_id"})
check("disabled Subcolor remembers its column without applying it",
      fake.session_state[vw.PICKER_COL_KEY] == "patient_id" and result[4] is None)

print("3. grouping exclusions and Collapse by stay consistent in all modes")
for mode in ("shape", "subcolor", "opacity"):
    label = f"{mode.title()} by"
    state = {vw.POINT_MODE_KEY: mode, "analysis_control_separate_by": "treatment",
             vw.COLOR_BY_KEY: ["experiment"], vw.PICKER_COL_KEY: "dish",
             vw.COLLAPSE_BY_KEY: "dish"}
    fake, result = run(state=state, collapse=True)
    check(f"{mode}: grouping excludes both grouping columns from decorations",
          not {"experiment", "treatment"}.intersection(fake.widget_options[label]))
    check(f"{mode}: the collapse column remains a decoration option",
          "dish" in fake.widget_options[label])
    check(f"{mode}: collapse leaves grouping choices available",
          "dish" in fake.widget_options["Separate by"]
          and "dish" in fake.widget_options[label_of(fake, "multiselect")])
    check(f"{mode}: collapse runs downstream of grouping",
          order_of(fake, "multiselect") < order_of(fake, "selectbox", "Collapse by")
          and not {"experiment", "treatment"}.intersection(fake.widget_options["Collapse by"]))
    for invalid in ("experiment", "treatment", "removed_column"):
        fake, result = run(state={**state, vw.PICKER_COL_KEY: invalid}, collapse=True)
        check(f"{mode}: invalid {invalid!r} clears safely",
              fake.session_state[vw.PICKER_COL_KEY] is None
              and all(result[i] is None for i in (1, 2, 4)))

fake, result = run(state={vw.COLOR_BY_KEY: ["dish"], vw.COLLAPSE_BY_KEY: "dish"}, collapse=True)
check("grouping on a collapse column retires collapse, preserving grouping",
      result[0] == ["dish"] and result[5] is None
      and fake.session_state[vw.COLLAPSE_BY_KEY] is None)
fake, result = run()
check("collapse is absent when not offered", "Collapse by" not in fake.widget_options and result[5] is None)

print("4. methods with independent decorations retain the existing behavior")
state = {vw.COLOR_BY_KEY: ["experiment"], vw.OPACITY_BY_KEY: "treatment",
         vw.PICKER_COL_KEY: "patient_id"}
fake, result = run(state=state, match=False)
check("shape and opacity can both be used", result[1:3] == ("treatment", "patient_id"), result)
fake, result = run(state={**state, vw.COLOR_BY_KEY: ["treatment"]}, match=False)
check("a newly grouped opacity column is cleared",
      result[1] is None and fake.session_state[vw.OPACITY_BY_KEY] is None)
check("a still-valid shape column survives", result[2] == "patient_id")

print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
