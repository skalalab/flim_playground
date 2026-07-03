"""Compute derived features: new columns built from arithmetic over existing ones.

A derived feature is defined by a dict::

    {"name": str, "expression": str, "operands": [column_name, ...]}

where ``expression`` uses positional aliases ``A, B, C, ...`` that map to
``operands[0], operands[1], ...``. Using aliases (rather than the real, often
messy, column names like ``"Lifetime fit_nadh: a1"``) keeps the stored formula
trivial to author and safe to parse.

Evaluation is a restricted-AST interpreter — NOT ``eval()``/``DataFrame.eval()`` —
so a formula can only reference its own aliases and use ``+ - * /`` and
parentheses. Anything else (function calls, attribute access, unknown names) is
rejected. Divide-by-zero yields NaN rather than raising or producing ``inf``.

The output column is named ``"Derived: {name}"``, which the analysis layer
(``src/dataset_io.py``, ``src/feature_labels.py``) recognises as a single
cross-channel "Derived Features" group.
"""
import ast

import numpy as np
import pandas as pd

# Binary operators the mini-language allows.
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def alias_names(count):
    """Positional aliases ``A, B, C, ...`` for *count* operands (A–Z, max 26)."""
    return [chr(ord("A") + i) for i in range(count)]


def _safe_divide(left, right):
    """Vectorized division with divide-by-zero -> NaN (never raises, keeps index)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = left / right
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    result = np.asarray(result, dtype="float64")
    return np.where(np.isinf(result), np.nan, result)


def _eval_node(node, series_by_alias):
    """Recursively evaluate a whitelisted AST node against an alias -> Series map."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, series_by_alias)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_node(node.left, series_by_alias)
        right = _eval_node(node.right, series_by_alias)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return _safe_divide(left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand, series_by_alias)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.Name):
        if node.id in series_by_alias:
            return series_by_alias[node.id]
        raise ValueError(f"unknown operand '{node.id}'")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    raise ValueError("only operands (A, B, …), numbers, + - * / and parentheses are allowed")


def evaluate_expression(expression, series_by_alias):
    """Parse and evaluate one alias expression against *series_by_alias*.

    Raises ``ValueError`` on any unsafe or malformed input (so callers can catch a
    single exception type).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid syntax: {exc.msg}") from exc
    return _eval_node(tree, series_by_alias)


def is_single_operand(expression):
    """True if *expression* is just a bare operand alias, e.g. ``"A"`` or ``"(B)"``.

    Such a "formula" merely duplicates an existing column, so the builder rejects
    it. Returns ``False`` for anything not parseable as a single bare name.
    """
    try:
        tree = ast.parse((expression or "").strip(), mode="eval")
    except SyntaxError:
        return False
    return isinstance(tree.body, ast.Name)


def compute_derived_features(df, derived_defs):
    """Append a ``"Derived: {name}"`` column to *df* for each definition.

    Returns ``(df, warnings)``. A definition is skipped (with a human-readable
    warning appended to *warnings*) when it is malformed, references a missing
    operand column, or has an invalid/unsafe expression — the function never
    raises, so one bad formula cannot abort a whole extraction run.
    """
    warnings = []
    if not derived_defs:
        return df, warnings

    for definition in derived_defs:
        name = (definition.get("name") or "").strip()
        expression = (definition.get("expression") or "").strip()
        operands = definition.get("operands", []) or []
        if not name or not expression:
            warnings.append("Skipped a derived feature with an empty name or expression.")
            continue

        # Map aliases to the operand columns that exist. Only operands actually
        # referenced by the expression matter, so we don't pre-reject on a
        # missing-but-unused operand; a referenced-yet-missing alias surfaces as
        # an "unknown operand" error below and is reported as a missing column.
        aliases = alias_names(len(operands))
        series_by_alias = {}
        missing = []
        for alias, col in zip(aliases, operands):
            if col in df.columns:
                series_by_alias[alias] = df[col]
            else:
                missing.append(col)

        try:
            result = evaluate_expression(expression, series_by_alias)
        except Exception as exc:
            if missing:
                warnings.append(
                    f"Derived feature '{name}' skipped: operand column(s) not found: {', '.join(missing)}."
                )
            else:
                warnings.append(f"Derived feature '{name}' skipped: {exc}.")
            continue

        df[f"Derived: {name}"] = result

    return df, warnings
