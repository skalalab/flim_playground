"""Shared FLIM feature-label vocabulary (co-design between extraction and analysis).

Single source of truth that turns the Data-Extraction column naming convention
("{Extractor}_{channel}: {feature}", plus uncategorized "{channel}_{suffix}") into
human-readable axis titles in proper FLIM notation: channel + Greek/scientific
symbol + unit, e.g. ``"nadh τ₁ (ps)"``.

``format_feature_label`` is imported by the Data-Analysis plots (``src/vis/*``) AND
inlined verbatim into exported Matplotlib scripts via ``inspect.getsource`` (see
``src/export_script.py``), so the GUI and the exported script always render identical
labels — there is no second copy of this mapping. For that reason the function is
fully self-contained: its lookup tables live inside it and it uses only the standard
library (``re``).
"""
import re


def format_feature_label(column_name):
    """Return a pretty FLIM axis label for an extracted feature column.

    Recognised columns become ``"{channel} {symbol} ({unit})"`` (the unit is omitted
    when the quantity is dimensionless). Unrecognised columns — identifiers,
    categoricals, or arbitrary user CSVs — are returned unchanged.

    Glyph convention: real unicode subscripts throughout (τ₁, α₂, τₘ, τᵩ, χ²ᵣ) —
    DejaVu Sans (Matplotlib's default) and common browser fonts include them. The sole
    exception is modulation lifetime τ_M: Unicode has no subscript capital M, and the
    capital M is what distinguishes it from the mean lifetime τₘ. Units verified
    against extracted data ranges: fit lifetimes are ps, phasor-derived lifetimes
    (Tau_*) are ns, amplitude fractions are %.
    """
    if not isinstance(column_name, str):
        return column_name

    # feature key (text after ": ") -> (symbol, unit). Empty unit => dimensionless.
    feature_labels = {
        # lifetime fit
        "t1": ("τ₁", "ps"), "t2": ("τ₂", "ps"), "t3": ("τ₃", "ps"),
        "tm": ("τₘ", "ps"), "tm_iw": ("τₘ,ᵢ", "ps"),
        "a1": ("α₁", "%"), "a2": ("α₂", "%"), "a3": ("α₃", "%"),
        # phasor coordinates (1st harmonic implicit)
        "G(1st)": ("g", ""), "S(1st)": ("s", ""),
        "G(2nd)": ("g (2nd harm.)", ""), "S(2nd)": ("s (2nd harm.)", ""),
        # phasor-derived lifetimes (ns)
        "Tau_phase": ("τᵩ", "ns"), "Tau_m": ("τ_M", "ns"),
        # morphology
        "area": ("Area", "px²"), "perimeter": ("Perimeter", "px"),
        "major_axis_length": ("Major axis", "px"),
        "minor_axis_length": ("Minor axis", "px"),
        "solidity": ("Solidity", ""), "eccentricity": ("Eccentricity", ""),
        "circularity": ("Circularity", ""),
        # texture
        "intensity_sum": ("Total intensity", "a.u."),
        "mass_displacement": ("Mass displacement", "px"),
    }
    # uncategorized suffix (text after "{channel}_") -> (symbol, unit)
    suffix_labels = {
        "amp1": ("A₁", "a.u."), "amp2": ("A₂", "a.u."), "amp3": ("A₃", "a.u."),
        "offset": ("Offset", "a.u."),
        "reduced_chi_square": ("χ²ᵣ", ""),
        "centroid_x": ("Centroid x", "px"), "centroid_y": ("Centroid y", "px"),
        "a2": ("α₂", "%"), "a3": ("α₃", "%"),
    }

    def compose(channel, symbol, unit):
        return f"{channel} {symbol} ({unit})" if unit else f"{channel} {symbol}"

    def lookup_feature(feature):
        """Map a feature key to (symbol, unit), handling parameterised families."""
        if feature in feature_labels:
            return feature_labels[feature]
        m = re.fullmatch(r"granularity_(\d+)", feature)
        if m:
            return (f"Granularity r={m.group(1)}", "%")
        m = re.fullmatch(r"radial_distribution_ring(\d+)", feature)
        if m:
            return (f"Radial dist. ring {m.group(1)}", "")
        return None

    # Categorised column: "{Extractor}_{channel}: {feature}"
    # (split mirrors src/dataset_io.py:get_feature_groups_data_extraction)
    if ": " in column_name:
        extractor_channel, feature = column_name.split(": ", 1)
        if "_" not in extractor_channel:
            return column_name
        channel = extractor_channel.split("_", 1)[1]
        mapped = lookup_feature(feature)
        if mapped is None:
            return column_name
        symbol, unit = mapped
        return compose(channel, symbol, unit)

    # Uncategorised column: "{channel}_{suffix}"
    for suffix, (symbol, unit) in suffix_labels.items():
        token = "_" + suffix
        if column_name.endswith(token) and len(column_name) > len(token):
            channel = column_name[: -len(token)]
            return compose(channel, symbol, unit)

    return column_name
