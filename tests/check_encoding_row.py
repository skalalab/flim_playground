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
        return [self._slot(f"col{i}") for i in range(self.n_cols)]

    def container(self, **kw):
        return self._slot(f"container{len(self.slots)}", **kw)

    def markdown(self, body, **kw):
        """Record visible text, excluding HTML tags and CSS."""
        text = re.sub(r"<style\b[^>]*>.*?</style>", "", body, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", "", text).strip()
        self.events.append(("markdown", text, self.stack[-1] if self.stack else None))
        self.visible.append(text)

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
        self.widget_kw[label] = kw
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
    fake.purge_unrendered([vw.PICKER_COL_KEY, vw.OPACITY_BY_KEY, vw.COLLAPSE_BY_KEY])
    return fake, result


def label_of(fake, kind):
    return next((lab for k, lab, _ in fake.events if k == kind), None)


def order_of(fake, kind, label=None):
    for i, (k, lab, _) in enumerate(fake.events):
        if k == kind and (label is None or lab == label):
            return i
    return -1


print("1. layout")
fake, _ = run()
check("four equal columns with Separate by", fake.n_cols == 4, fake.n_cols)
fake3, _ = run(separate=False)
check("three equal columns without Separate by", fake3.n_cols == 3, fake3.n_cols)
fakep, _ = run(match=False)
check("unchanged for methods with no colour channel", fakep.n_cols == 4, fakep.n_cols)
fakec, _ = run(collapse=True)
check("five with Collapse by", fakec.n_cols == 5, fakec.n_cols)
check("Collapse by is DISPLAYED third, after the two grouping channels",
      [lab for k, lab, slot in fakec.events
       if k == "selectbox" and slot == "col2"] == ["Collapse by"],
      [(k, lab, slot) for k, lab, slot in fakec.events if k == "selectbox"])

print("2. evaluation order vs display order")
fake, _ = run(state={vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"],
                     vw.PICKER_COL_KEY: "patient_id"})
check("the switch's slot evaluated before the colour multiselect",
      order_of(fake, "selectbox", vw.PICKER_LABELS[True]) < order_of(fake, "multiselect"),
      fake.events)
check("switch evaluated before the picker it labels",
      order_of(fake, "toggle") < order_of(fake, "selectbox", vw.PICKER_LABELS[True]),
      fake.events)
label_slot = next(w for k, _lab, w in fake.events if k == "markdown")
switch_slot = next(w for k, _lab, w in fake.events if k == "toggle")
# The label and switch share a content-width horizontal row.
check("switch shares one row with the hand-drawn label",
      fake.slot_parent.get(label_slot) == switch_slot,
      f"label in {label_slot} (parent {fake.slot_parent.get(label_slot)}), "
      f"switch in {switch_slot}")
check("that row lays out horizontally",
      fake.slot_kw.get(switch_slot, {}).get("horizontal") is True,
      fake.slot_kw.get(switch_slot))
check("neither label nor switch stretches, so they stay adjacent",
      fake.slot_kw.get(label_slot, {}).get("width") == "content"
      and fake.widget_kw.get(vw.SWITCH_TRAIL, {}).get("width") == "content",
      f"label {fake.slot_kw.get(label_slot)}, "
      f"switch {fake.widget_kw.get(vw.SWITCH_TRAIL)}")
check("the picker itself is not in that row, so it keeps the full column width",
      fake.slot_parent.get(next(w for k, lab, w in fake.events
                                if k == "selectbox"
                                and lab == vw.PICKER_LABELS[True])) != switch_slot)
# The phrase stays fixed while the switch selects the active channel.
drawn = lambda f: next(lab for k, lab, _w in f.events if k == "markdown")
fake_off, _ = run(state={vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"]})
for state_name, f in (("switch on", fake), ("switch off", fake_off)):
    check(f"{state_name}: phrase reads '{vw.SWITCH_LEAD} [switch] {vw.SWITCH_TRAIL}'",
          drawn(f) == vw.SWITCH_LEAD
          and any(k == "toggle" and lab == vw.SWITCH_TRAIL for k, lab, _w in f.events),
          f"drawn {drawn(f)!r}, toggle "
          f"{[lab for k, lab, _w in f.events if k == 'toggle']}")
check("the phrase does not change with the switch", drawn(fake) == drawn(fake_off),
      f"{drawn(fake)!r} vs {drawn(fake_off)!r}")
# Container position determines display order independently of evaluation order.
def column_of(f, kind, label=None):
    """Find a widget's display column through nested containers."""
    slot = next(w for k, lab, w in f.events
                if k == kind and (label is None or lab == label))
    while slot is not None and not slot.startswith("col"):
        slot = f.slot_parent.get(slot)
    return slot
check("the switch's picker is in the last column, Opacity by in the one before",
      column_of(fake, "selectbox", vw.PICKER_LABELS[True]) == "col3"
      and column_of(fake, "selectbox", "Opacity by") == "col2",
      f"switch picker {column_of(fake, 'selectbox', vw.PICKER_LABELS[True])}, "
      f"opacity {column_of(fake, 'selectbox', 'Opacity by')}")

print("3. exactly one control offers the colour channel")
# Either the Color by multiselect or the enabled switch controls colour.
for name, state in {
    "switch off": {vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"]},
    "switch on, nothing picked": {vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"]},
    "switch on, column picked": {vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"],
                                 vw.PICKER_COL_KEY: "patient_id"},
    "switch on, no groups": {vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: []},
}.items():
    fake, _ = run(state=state)
    multiselect_claims = label_of(fake, "multiselect") == "Color by"
    switch_claims = bool(state.get(vw.AS_COLOUR_KEY))
    check(f"{name}: exactly one claim on colour",
          multiselect_claims != switch_claims,
          f"multiselect={'Color by' if multiselect_claims else label_of(fake, 'multiselect')}, "
          f"switch={'on' if switch_claims else 'off'} -- visible {fake.visible}")
    # Ignore collapsed labels when checking for duplicate visible control names.
    dupes = {lab for lab in fake.visible if fake.visible.count(lab) > 1}
    check(f"{name}: no two visible controls share a name", not dupes,
          f"duplicated {sorted(dupes)} -- {fake.visible}")

print("4. the relabel happens in the same run")
fake, (_c, _o, _s, _sep, subcolor_by, _collapse) = run(
    state={vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"],
           vw.PICKER_COL_KEY: "patient_id"})
check("first picker is 'Group by' once the switch claims colour",
      label_of(fake, "multiselect") == "Group by", label_of(fake, "multiselect"))
check("subcolor_by is returned", subcolor_by == "patient_id", subcolor_by)
check("opacity_by is not also set", _o is None, _o)

fake, _ = run(state={vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"]})
check("renamed even before a column is picked",
      label_of(fake, "multiselect") == "Group by", label_of(fake, "multiselect"))

fake, _ = run(state={vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"],
                     vw.PICKER_COL_KEY: "patient_id"})
check("reverts with the switch off", label_of(fake, "multiselect") == "Color by",
      label_of(fake, "multiselect"))
check("opacity picker is shown instead",
      any(lab == "Opacity by" for _k, lab, _w in fake.events), fake.events)

fake, (_c, _o, _s, _sep, subcolor_by, _collapse) = run(
    state={vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"],
           vw.PICKER_COL_KEY: "experiment"})
check("a match column claimed by Group by is pruned away", subcolor_by is None, subcolor_by)

print("5. grouping columns are struck from EVERY decoration")
# Decorations exclude columns used by either grouping control.
fake, (color_by, _o, _s, sep, subcolor_by, _collapse) = run(
    state={"analysis_control_separate_by": "experiment",
           vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["treatment"],
           vw.PICKER_COL_KEY: "patient_id"})
check("Separate by is returned", sep == "experiment", sep)
# Inspect offered options; an unselected return value cannot prove exclusion.
check("a column used by Separate by is not offered as a colour group",
      "experiment" not in fake.widget_options.get("Group by", []),
      fake.widget_options.get("Group by"))
check("nor is it offered as the subcolor column",
      "experiment" not in fake.widget_options.get(vw.PICKER_LABELS[True], []),
      fake.widget_options.get(vw.PICKER_LABELS[True]))
check("nor is a column Color by is grouping on",
      "treatment" not in fake.widget_options.get(vw.PICKER_LABELS[True], []),
      fake.widget_options.get(vw.PICKER_LABELS[True]))

# Shape follows the same exclusions as subcolor.
fake, (_c, _o, shape_by, sep, _m, _collapse) = run(
    state={"analysis_control_separate_by": "treatment",
           vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"],
           vw.PICKER_COL_KEY: "treatment"})
check("the Shape role obeys the same rule",
      "treatment" not in fake.widget_options.get(vw.PICKER_LABELS[False], []),
      fake.widget_options.get(vw.PICKER_LABELS[False]))
check("and cannot hold it", shape_by is None, shape_by)
check("nor is the colour column offered to it",
      "experiment" not in fake.widget_options.get(vw.PICKER_LABELS[False], []),
      fake.widget_options.get(vw.PICKER_LABELS[False]))

# Opacity, the third: its own column on every point-based method, and the same list.
fake, _ = run(state={"analysis_control_separate_by": "treatment",
                     vw.COLOR_BY_KEY: ["experiment"]})
check("Opacity by excludes both as well",
      not {"treatment", "experiment"} & set(fake.widget_options.get("Opacity by", [])),
      fake.widget_options.get("Opacity by"))

# A changing option list preserves valid keyed selections and clears invalid ones.
fake, (_c, opacity_by, _s, _sep, _m, _collapse) = run(
    state={vw.COLOR_BY_KEY: ["experiment"], vw.OPACITY_BY_KEY: "treatment"})
check("a pick survives a change to Color by that does not touch it",
      opacity_by == "treatment", opacity_by)

fake, (_c, opacity_by, _s, _sep, _m, _collapse) = run(
    state={vw.COLOR_BY_KEY: ["treatment"], vw.OPACITY_BY_KEY: "treatment"})
check("grouping ON the held column retires it", opacity_by is None, opacity_by)
check("and clears it from session state, so the keyed widget cannot raise on it",
      fake.session_state.get(vw.OPACITY_BY_KEY) is None,
      fake.session_state.get(vw.OPACITY_BY_KEY))

print("6. one selection, shared by both roles")
# Both switch positions use the latest selected column.
state = {vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"]}
fake, _ = run(state=state)
state = dict(fake.session_state)
state[vw.PICKER_COL_KEY] = "treatment"          # picked while the switch is off
fake, (_c, _o, shape_by, _sep, _m, _collapse) = run(state=state)
check("held as shape while the switch is off", shape_by == "treatment", shape_by)

state = dict(fake.session_state); state[vw.AS_COLOUR_KEY] = True
fake, (_c, _o, shape_by, _sep, subcolor_by, _collapse) = run(state=state)
check("the SAME column carries over to subcolor", subcolor_by == "treatment", subcolor_by)
check("and stops driving shape", shape_by is None, shape_by)

state = dict(fake.session_state); state[vw.AS_COLOUR_KEY] = False
fake, (_c, _o, shape_by, _sep, subcolor_by, _collapse) = run(state=state)
check("and carries back again", shape_by == "treatment", shape_by)
check("without also driving subcolor", subcolor_by is None, subcolor_by)

# A new subcolor selection must also become the shape selection.
state = dict(fake.session_state); state[vw.AS_COLOUR_KEY] = True
state[vw.PICKER_COL_KEY] = "patient_id"         # changed while on subcolor
fake, (_c, _o, _s, _sep, subcolor_by, _collapse) = run(state=state)
check("a change made on subcolor sticks", subcolor_by == "patient_id", subcolor_by)
state = dict(fake.session_state); state[vw.AS_COLOUR_KEY] = False
fake, (_c, _o, shape_by, _sep, _m, _collapse) = run(state=state)
check("flipping back shows the NEW column, not the old one",
      shape_by == "patient_id", shape_by)

# Clear selections that no longer appear in the options.
state = {vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"],
         vw.PICKER_COL_KEY: "experiment"}
fake, (_c, _o, _s, _sep, subcolor_by, _collapse) = run(state=state)
check("a column Color by is grouping on cannot also subdivide it",
      subcolor_by is None, subcolor_by)

state = {vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"],
         vw.PICKER_COL_KEY: "not_a_column"}
fake, (_c, _o, shape_by, _sep, _m, _collapse) = run(state=state)
check("a column that is no longer offered is dropped", shape_by is None, shape_by)
# Check state too: the stub's filtered return alone cannot prove the value was cleared.
check("and cleared from session state, so the keyed widget cannot raise on it",
      fake.session_state.get(vw.PICKER_COL_KEY) is None,
      fake.session_state.get(vw.PICKER_COL_KEY))

print("7. Collapse by is LAST in the grouping chain")
# Grouping constrains Collapse by; collapsing must leave grouping options intact.
state = {vw.COLLAPSE_BY_KEY: "dish", vw.AS_COLOUR_KEY: False,
         vw.COLOR_BY_KEY: ["experiment"]}
fake, (_c, _o, _s, _sep, _sub, collapse_by) = run(state=state, collapse=True)
check("the picked column is returned", collapse_by == "dish", collapse_by)
check("Color by keeps its full list", "dish" in fake.widget_options["Color by"],
      fake.widget_options["Color by"])
check("Separate by keeps its full list", "dish" in fake.widget_options["Separate by"],
      fake.widget_options["Separate by"])
check("a grouped column is struck from Collapse by",
      "experiment" not in fake.widget_options["Collapse by"],
      fake.widget_options["Collapse by"])
check("evaluated AFTER the colour multiselect",
      order_of(fake, "multiselect") < order_of(fake, "selectbox", "Collapse by"),
      fake.events)

# The collapse column remains available to decorate each replicate's point.
for as_colour, label in ((False, vw.PICKER_LABELS[False]), (True, vw.PICKER_LABELS[True])):
    state = {vw.COLLAPSE_BY_KEY: "dish", vw.AS_COLOUR_KEY: as_colour,
             vw.COLOR_BY_KEY: ["experiment"]}
    fake, _ = run(state=state, collapse=True)
    check(f"still offered as {label}", "dish" in fake.widget_options[label],
          fake.widget_options[label])
check("still offered as Opacity by", "dish" in fake.widget_options["Opacity by"],
      fake.widget_options["Opacity by"])

# The yield goes the other way: grouping on the collapsed column retires the collapse.
state = {vw.COLLAPSE_BY_KEY: "dish", vw.AS_COLOUR_KEY: False,
         vw.COLOR_BY_KEY: ["dish"]}
fake, (_c, _o, _s, _sep, _sub, collapse_by) = run(state=state, collapse=True)
check("grouping ON the collapsed column retires the collapse, not the grouping",
      collapse_by is None, collapse_by)
check("and cleared from session state, so the keyed widget cannot raise on it",
      fake.session_state.get(vw.COLLAPSE_BY_KEY) is None,
      fake.session_state.get(vw.COLLAPSE_BY_KEY))

fake, (_c, _o, _s, _sep, _sub, collapse_by) = run()
check("absent when the method does not offer it", collapse_by is None, collapse_by)
check("and not drawn", "Collapse by" not in fake.visible, fake.visible)

print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
