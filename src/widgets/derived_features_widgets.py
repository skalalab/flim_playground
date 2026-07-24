"""Compact Configuration-page editor for per-profile derived features.

Revealed by a "Derived features" checkbox on the (already large) Configuration
page (``main.py``), so it must stay self-contained and small. It reads the current
list from the active-profile ``cfg`` for display, builds operand choices from the
profile's *live* extractor selections (``predict_feature_columns_from_cfg``), and
persists Add/Delete immediately via ``set_derived_features`` — mirroring the
profile create/delete pattern — so an edit survives the Streamlit rerun without
waiting for the page-level "Update Configuration" save.

See ``src/derived_features.py`` for the storage schema and safe evaluator.
"""
import re

import pandas as pd
import streamlit as st

from src.config import set_derived_features
from src.derived_features import alias_names, evaluate_expression, is_single_operand
from src.emojis import happy_emoji, sad_emoji
from src.feature_schema import predict_feature_columns_from_cfg

# Shown as the help "?" on the reveal checkbox in main.py (st.expander has no help
# parameter in Streamlit 1.54, so the checkbox carries this description instead).
DERIVED_FEATURES_HELP = (
    "Build new features from arithmetic over already-extracted features "
    "(e.g. a redox ratio). Each becomes a **Derived: _name_** column and, in "
    "Data Analysis, a single **Derived Features** group. Operands may span "
    "channels."
)

# Template label -> {expression, arity}. "Custom…" (None) unlocks the free editor.
_TEMPLATES = {
    "Normalized  A / (A + B)": {"expression": "A/(A+B)", "n": 2},
    "Ratio  A / B": {"expression": "A/B", "n": 2},
    "Difference  A − B": {"expression": "A-B", "n": 2},
    "Sum  A + B": {"expression": "A+B", "n": 2},
    "Custom…": None,
}

# A lone +, -, or * is a Markdown bullet-list marker, so st.button (which renders
# its label as Markdown) shows it blank. Escape the label; the *raw* operator is
# still what gets appended to the formula. /, (, ) aren't list markers -> fine.
_OP_LABELS = {"+": r"\+", "-": r"\-", "*": r"\*"}


def _append_token(state_key, token):
    """on_click callback: append *token* to the formula box's session_state value."""
    st.session_state[state_key] = (st.session_state.get(state_key, "") or "") + token


def _expand_expression(expression, operands):
    """Substitute aliases with their operand column names, for display only."""
    def repl(match):
        idx = ord(match.group(0)) - ord("A")
        return operands[idx] if idx < len(operands) else match.group(0)
    return re.sub(r"[A-Z]", repl, expression)


def _validate(expression, operands):
    """Return (is_valid, error) by evaluating the formula against dummy data."""
    expression = (expression or "").strip()  # validate the form that gets stored
    if not expression:
        return False, "empty formula"
    if is_single_operand(expression):
        return False, "use an operator — a lone feature (e.g. 'A') just duplicates a column"
    dummy = {alias: pd.Series([1.0, 2.0, 3.0]) for alias in alias_names(len(operands))}
    try:
        evaluate_expression(expression, dummy)
    except Exception as exc:
        return False, str(exc)
    return True, ""


def render_derived_features_widget(cfg, active_profile):
    """Render the derived-features editor for the active profile.

    *cfg* is the active profile's config sub-dict (used for display + operand
    prediction); writes go straight to disk via ``set_derived_features``. The
    section-level description lives on the reveal checkbox in main.py
    (``DERIVED_FEATURES_HELP``), not here.
    """
    # Every builder INPUT widget is keyed by a per-profile "generation" counter
    # that the Add handler bumps. A fresh generation gives those widgets brand-new
    # keys on the next run, so they come up empty/default — the only reliable reset
    # in the live app. (Deleting a widget's session_state key leaves its identity
    # unchanged, so Streamlit restores the previous *frontend* value on the rerun:
    # AppTest, which has no frontend, resets fine, but the live app does not — the
    # bug this replaces.) The Template selectbox is deliberately NOT generation-
    # scoped, so the chosen template persists across successive adds.
    gen = st.session_state.setdefault(f"df_gen_{active_profile}", 0)

    existing = cfg.get("derived_features", []) or []

    # --- Existing derived features: expanded formula + a delete button per row.
    if existing:
        for i, definition in enumerate(existing):
            name = definition.get("name", "")
            expr = definition.get("expression", "")
            operands = definition.get("operands", [])
            row = st.columns([0.85, 0.15], vertical_alignment="center")
            with row[0]:
                st.markdown(f"**{name}**  =  `{_expand_expression(expr, operands)}`")
            with row[1]:
                if st.button("🗑️", key=f"df_del_{active_profile}_{i}", help=f"Delete '{name}'"):
                    set_derived_features([d for j, d in enumerate(existing) if j != i])
                    st.rerun()
    else:
        st.caption("No derived features yet. Add one below.")

    # --- Operand choices come from the profile's *current* extractor selections.
    available = predict_feature_columns_from_cfg(cfg)
    if not available:
        st.warning(
            "Select feature extractors for your channels first — there are no "
            "features to derive from yet."
        )
        return

    tcol, ncol = st.columns([1, 1])
    template_label = tcol.selectbox(
        "Template", list(_TEMPLATES.keys()), key=f"df_template_{active_profile}",
        help="Start from a common formula, or choose Custom… to compose your own.",
    )
    template = _TEMPLATES[template_label]

    name = ncol.text_input(
        "Name", key=f"df_name_{active_profile}_{gen}", placeholder="e.g. redox_ratio",
        help="The new column will be named 'Derived: <name>'.",
    )

    if template is not None:
        # Fixed-arity template: one searchable dropdown per alias slot (A, B, …).
        aliases = alias_names(template["n"])
        operands = []
        slot_cols = st.columns(template["n"])
        for slot, alias in enumerate(aliases):
            with slot_cols[slot]:
                operands.append(st.selectbox(
                    f"{alias}", available, key=f"df_op_{active_profile}_{alias}_{gen}",
                ))
        expression = template["expression"]
        st.caption(f"Formula:  `{expression}`")
    else:
        # Custom: pick any number of operands (aliased A, B, C… in pick order),
        # then compose a free expression with a text box + append-buttons.
        operands = st.multiselect(
            "Operands (each becomes A, B, C… in the order picked)",
            available, key=f"df_ops_{active_profile}_{gen}",
        )
        alias_hint = ",  ".join(
            f"{a} = {c}" for a, c in zip(alias_names(len(operands)), operands)
        )
        if alias_hint:
            st.caption(alias_hint)

        expr_key = f"df_expr_{active_profile}_{gen}"
        st.session_state.setdefault(expr_key, "")

        # Append-buttons: operators, then one per available alias. Streamlit
        # text_input exposes no caret position, so buttons append at the end.
        op_cols = st.columns(6)
        for col, token in zip(op_cols, ["+", "-", "*", "/", "(", ")"]):
            with col:
                st.button(_OP_LABELS.get(token, token),
                          key=f"df_btn_{active_profile}_{token}",
                          width="stretch",
                          on_click=_append_token, args=(expr_key, token))
        if operands:
            alias_cols = st.columns(len(operands))
            for col, alias in zip(alias_cols, alias_names(len(operands))):
                with col:
                    st.button(alias, key=f"df_btn_{active_profile}_{alias}",
                              width="stretch",
                              on_click=_append_token, args=(expr_key, alias))

        expression = st.text_input(
            "Formula", key=expr_key,
            help="Use aliases (A, B, …), + − * /, and parentheses. Buttons append "
                 "to the end; you can also edit the box directly.",
        )

    # --- Live validation + preview.
    name_clean = name.strip()
    name_ok = bool(name_clean) and ": " not in name_clean
    valid, err = _validate(expression, operands)
    duplicate = any(d.get("name") == name_clean for d in existing)

    if name and not name_ok:
        st.error(f"Name cannot contain ': '. {sad_emoji}")
    if name_clean and expression:
        if not valid:
            st.error(f"Invalid formula: {err} {sad_emoji}")
        elif duplicate:
            st.caption(f":red[A derived feature named '{name_clean}' already exists.]")
        else:
            st.success(f"✓  `Derived: {name_clean}`  =  `{_expand_expression(expression, operands)}` {happy_emoji}")

    # --- Add.
    disabled = not (name_ok and expression and valid and operands and not duplicate)
    if st.button("➕ Add derived feature", key=f"df_add_{active_profile}", disabled=disabled):
        set_derived_features(existing + [{
            "name": name_clean,
            "expression": expression.strip(),
            "operands": list(operands),
        }])
        # Bump the generation so every input widget is recreated with a fresh key
        # (hence empty/default) on the next run. See the note at the top of this
        # function for why deleting the keys doesn't reset them in the live app.
        st.session_state[f"df_gen_{active_profile}"] = gen + 1
        st.rerun()
