"""The visual-encoding row: layout, evaluation order, and the "Group by" relabel.

Three things the widget has to get right, none of which any other check covers:

  1. Equal column widths. The channel switch is absorbed inside the third column
     rather than given a column of its own, so this row must lay out identically to
     Scatter/Phasor/UMAP, which have no switch at all.
  2. The third slot is EVALUATED before the Color by multiselect even though it
     DISPLAYS to its right; the switch inside it is evaluated before that
     multiselect, and displays above the picker rather than beside it, on the line the
     drawn label shares. It relies on Streamlit containers being fillable out of order.
  3. The multiselect relabels to "Group by" in the SAME run the colour channel is
     taken -- not a run later.

Streamlit is stubbed rather than driven: AppTest cannot reach the analysis page (no
file_uploader accessor), and bare mode returns widget defaults and ignores
session_state, which is exactly the state this needs to control.

    python tests/check_encoding_row.py
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
    def __init__(self, state, answers):
        self.n_cols = None
        self.session_state = state
        self.answers = answers      # {label: value the widget should return}
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
        # Parentage and kwargs are recorded because the switch now shares the picker's
        # top line: "beside, not under" is a claim about which container each was
        # written into and how that container lays out, which slot order alone cannot
        # distinguish from the old stacked layout.
        self.slots.append(name)
        self.slot_kw[name] = kw
        self.slot_parent[name] = self.stack[-1] if self.stack else None
        return Slot(self, name)

    def columns(self, spec, **kw):
        # st.columns takes either a count or a per-column weight list. The widget passes
        # a count now that every column is equal; accept both so the stub does not quietly
        # constrain which one it may use.
        self.n_cols = spec if isinstance(spec, int) else len(spec)
        return [self._slot(f"col{i}") for i in range(self.n_cols)]

    def container(self, **kw):
        return self._slot(f"container{len(self.slots)}", **kw)

    def markdown(self, body, **kw):
        """The widget draws the third picker's label itself so the switch can sit right
        after it. Tags are stripped because what is checked is the text a user reads,
        not the styling carrying it."""
        # A <style> block renders nothing, so its CSS is not text the user reads -- drop
        # the block whole before stripping tags, or the rule that sizes the switch's label
        # arrives as part of the label.
        text = re.sub(r"<style\b[^>]*>.*?</style>", "", body, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", "", text).strip()
        self.events.append(("markdown", text, self.stack[-1] if self.stack else None))
        self.visible.append(text)

    def _widget(self, kind, label, key, default, label_visibility="visible"):
        """Streamlit's keyed-widget contract: for a widget with ``key``, session state
        already holds what the widget will return before it renders, so a run can never
        see the two disagree. Modelling that is the point -- letting ``answers`` override
        a keyed return independently would test a state the real thing cannot reach.
        ``answers`` therefore only serves the unkeyed pickers (Separate by, Shape by)."""
        where = self.stack[-1] if self.stack else None
        self.events.append((kind, label, where))
        if key is not None:
            self.rendered_keys.add(key)
        # A collapsed label is still handed to the widget -- screen readers read it, and
        # the multiselect checks itself against it -- but the user cannot see it, so it
        # does not count towards "exactly one control named Color by".
        if label_visibility != "collapsed":
            self.visible.append(label)
        if key is not None:
            value = self.session_state.get(key, default)
            self.session_state[key] = value
            return value
        return self.answers.get(label, default)

    def selectbox(self, label, options, index=None, key=None, disabled=False,
                  label_visibility="visible", **kw):
        self.widget_kw[label] = kw
        self.widget_options[label] = list(options)
        # `disabled` is accepted and ignored on purpose: a disabled selectbox still
        # returns what it last held, which is why the widget forces subcolor_by to None
        # separately rather than relying on the flag.
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
        """Streamlit drops the state of a keyed widget that did not render this run.

        By KEY, which is what Streamlit purges by and the only identity that survives the
        row's aliasing: the shape/subcolor picker answers to ONE key through two labels and
        is purged only if neither rendered -- the property that shared key buys -- so a
        label-based model could not express "not purged" for it at all.
        """
        for key in keys:
            if key not in self.rendered_keys:
                # Set to None rather than delete: observed against a live Streamlit,
                # and it is the stricter of the two -- a restore written as
                # .get(key, fallback) passes when the key is deleted and fails when it
                # is None, so the weaker stub would hide that.
                self.session_state[key] = None


DF = pd.DataFrame({
    "experiment": ["E1", "E2"] * 4,
    "patient_id": ["P1", "P2", "P3", "P4"] * 2,
    "treatment": ["T1", "T2"] * 4,
    "dish": ["d1", "d2"] * 4,
})
CATS = ["experiment", "patient_id", "treatment", "dish"]


def run(state=None, answers=None, separate=True, match=True, collapse=False):
    fake = FakeStreamlit(dict(state or {}), dict(answers or {}))
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
# The switch sits immediately after the picker's LABEL, which is a claim about which
# container each was written into -- not about slot order, which cannot tell this apart
# from the old stacked layout. The label is drawn by hand into a content-width child of a
# horizontal row and the switch straight into that same row, also content-width, so the
# two sit adjacent and left-aligned instead of the switch being pushed to the far edge.
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
# The phrase is STATIC: both channels stay on screen and the knob says which is live.
# So the halves must not drift with the switch -- a label that changed would be the old
# relabelling design creeping back, and would leave the phrase reading "color by color by".
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
# Display position, not evaluation order: the switch's slot is last on screen but is
# evaluated first, and Opacity by is now the one evaluated last.
def column_of(f, kind, label=None):
    """The COLUMN a widget ended up in, walking out through any nested containers --
    the switch's picker sits inside two of them."""
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
# The invariant is about the CLAIM, not the wording. It used to be checkable as "exactly
# one control is named Color by", but the third slot no longer spells that on screen --
# it shows a static phrase and lets the knob pick a half. So count claims: the first
# multiselect claims colour by being named "Color by", and the switch claims it by being
# on. Exactly one may claim it, or the user cannot tell which control drives the colours.
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
    # What stops the picker colliding with the multiselect is that its label is
    # COLLAPSED -- `visible` excludes collapsed labels -- with the rename to "Subcolor by"
    # as belt and braces. So guard the general property rather than that one pair: no two
    # controls on screen may carry the same name, whichever names they end up with.
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
# A decoration marks a point inside the slot its grouping put it in, so a column already
# spent on Separate by or Color by would give every point in a slot the same mark. One
# rule for all three channels, where it used to apply to subcolor alone.
# The `answers` mechanism exists for the unkeyed pickers but nothing used it, so
# separate_by was None in every case above and this exclusion -- the one cross-control
# rule in the row -- went unchecked.
fake, (color_by, _o, _s, sep, subcolor_by, _collapse) = run(
    answers={"Separate by": "experiment"},
    state={vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["treatment"],
           vw.PICKER_COL_KEY: "patient_id"})
check("Separate by is returned", sep == "experiment", sep)
# On the OPTIONS, not on the returned value: the value is whatever was stored, so it can
# exclude the Separate by column for reasons that have nothing to do with this rule. Only
# the offered list shows the rule being applied.
check("a column used by Separate by is not offered as a colour group",
      "experiment" not in fake.widget_options.get("Group by", []),
      fake.widget_options.get("Group by"))
check("nor is it offered as the subcolor column",
      "experiment" not in fake.widget_options.get(vw.PICKER_LABELS[True], []),
      fake.widget_options.get(vw.PICKER_LABELS[True]))
check("nor is a column Color by is grouping on",
      "treatment" not in fake.widget_options.get(vw.PICKER_LABELS[True], []),
      fake.widget_options.get(vw.PICKER_LABELS[True]))

# The Shape role obeys it too. It used to be the exception -- it was offered every
# category, on the reasoning that only the colour side competes with Separate by -- but a
# shape that never varies inside an x slot says nothing either, and having two decorations
# answer to different rules was the inconsistency this closed.
fake, (_c, _o, shape_by, sep, _m, _collapse) = run(
    answers={"Separate by": "treatment"},
    state={vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"],
           vw.PICKER_COL_KEY: "treatment"})
check("the Shape role obeys the same rule",
      "treatment" not in fake.widget_options.get(vw.PICKER_LABELS[False], []),
      fake.widget_options.get(vw.PICKER_LABELS[False]))
check("and cannot hold it", shape_by is None, shape_by)
check("nor is the colour column offered to it",
      "experiment" not in fake.widget_options.get(vw.PICKER_LABELS[False], []),
      fake.widget_options.get(vw.PICKER_LABELS[False]))

# Opacity, the third: its own column on every point-based method, and the same list.
fake, _ = run(answers={"Separate by": "treatment"},
              state={vw.COLOR_BY_KEY: ["experiment"]})
check("Opacity by excludes both as well",
      not {"treatment", "experiment"} & set(fake.widget_options.get("Opacity by", [])),
      fake.widget_options.get("Opacity by"))

# Narrowing that list is what forces Opacity by to be KEYED. An unkeyed widget is
# identified by its arguments, options included, so a list that moves for a reason having
# nothing to do with the held column used to remount this picker and blank it -- measured
# on the live page, with the dropped column still sitting in the list. The key buys the
# survival; the prune is what it costs, since Streamlit raises on a stored value a keyed
# widget no longer offers.
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
# The switch changes which channel the column drives, not which column. So a flip must
# leave the selection exactly where it was -- and must NOT resurrect whatever that role
# held on some earlier flip: pick A as shape, flip to subcolor, pick B, flip back, and A
# must not come back instead of B.
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

# The regression the shared key exists to kill: change the column in one role, and the
# other role must see the NEW one, not the one it held before.
state = dict(fake.session_state); state[vw.AS_COLOUR_KEY] = True
state[vw.PICKER_COL_KEY] = "patient_id"         # changed while on subcolor
fake, (_c, _o, _s, _sep, subcolor_by, _collapse) = run(state=state)
check("a change made on subcolor sticks", subcolor_by == "patient_id", subcolor_by)
state = dict(fake.session_state); state[vw.AS_COLOUR_KEY] = False
fake, (_c, _o, shape_by, _sep, _m, _collapse) = run(state=state)
check("flipping back shows the NEW column, not the old one",
      shape_by == "patient_id", shape_by)

# Two cases where a flip legitimately does clear it, both because the value is not on
# offer and an explicit key makes that a raise rather than a silent reset.
state = {vw.AS_COLOUR_KEY: True, vw.COLOR_BY_KEY: ["experiment"],
         vw.PICKER_COL_KEY: "experiment"}
fake, (_c, _o, _s, _sep, subcolor_by, _collapse) = run(state=state)
check("a column Color by is grouping on cannot also subdivide it",
      subcolor_by is None, subcolor_by)

state = {vw.AS_COLOUR_KEY: False, vw.COLOR_BY_KEY: ["experiment"],
         vw.PICKER_COL_KEY: "not_a_column"}
fake, (_c, _o, shape_by, _sep, _m, _collapse) = run(state=state)
check("a column that is no longer offered is dropped", shape_by is None, shape_by)
# On the SESSION STATE, not just the return: the stub filters an unoffered value to None
# (see selectbox above) exactly where real Streamlit raises, so the returned value alone
# passes whether or not the widget pruned. The write-back is the observable effect.
check("and cleared from session state, so the keyed widget cannot raise on it",
      fake.session_state.get(vw.PICKER_COL_KEY) is None,
      fake.session_state.get(vw.PICKER_COL_KEY))

print("7. Collapse by is LAST in the grouping chain")
# Separate by narrows Color by; the two of them narrow Collapse by. The direction is the
# point: collapsing is DERIVED from the x layout, so changing the layout may retire a
# collapse column, but changing the collapse must leave the layout alone -- reading it
# first inverted that, and picking a replicate silently reset the grouping.
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

# The collapse column is NOT struck from the decorations, in any role: a decoration is
# well defined on a collapsed dot whenever its column is constant within the collapse
# group, and the collapse column trivially is. Subcolor by = Collapse by is the SuperPlot.
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
