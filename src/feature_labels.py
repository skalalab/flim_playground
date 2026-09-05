"""Shared FLIM axis labels for analysis plots and exported Matplotlib scripts.

Convert extracted column names to channel, symbol, and unit labels such as
``"nadh τ₁ (ps)"``. Keep format_feature_label self-contained apart from re:
export_script.py embeds it with inspect.getsource(), including its lookup tables.
"""
import re


def format_feature_label(column_name, engine="plotly"):
    """Return a FLIM axis label for an extracted feature column.

    Recognised columns become ``"{channel} {symbol} ({unit})"`` (the unit is omitted
    when the quantity is dimensionless). Unrecognised columns — identifiers,
    categoricals, or arbitrary user CSVs — are returned unchanged.

    ``engine="plotly"`` uses HTML for subscripts unavailable in Unicode;
    ``engine="mpl"`` converts that markup to mathtext. Unicode symbols are shared.
    Modulation lifetime ``Tau_mod`` (alias ``Tau_m``) uses τ_mod, distinct from
    mean lifetime ``tm`` (τₘ). Fit lifetimes are ps, phasor-derived lifetimes are
    ns, and amplitude fractions are %.
    """
    if not isinstance(column_name, str):
        return column_name

    # Derived features carry no fixed unit/notation; show the user-given name.
    if column_name.startswith("Derived: "):
        return column_name.split(": ", 1)[1]

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
        "Tau_phase": ("τᵩ", "ns"), "Tau_mod": ("τ<sub>mod</sub>", "ns"),
        "Tau_m": ("τ<sub>mod</sub>", "ns"),  # Supported alias for Tau_mod.
        # morphology
        "area": ("Area", "px²"), "perimeter": ("Perimeter", "px"),
        "major_axis_length": ("Major axis", "px"),
        "minor_axis_length": ("Minor axis", "px"),
        "solidity": ("Solidity", ""), "eccentricity": ("Eccentricity", ""),
        "circularity": ("Circularity", ""),
        # texture
        "intensity_sum": ("Intensity", "photons"),
        "mass_displacement": ("Mass displacement", "px"),
    }
    # uncategorized suffix (text after "{channel}_") -> (symbol, unit)
    suffix_labels = {
        "amp1": ("A₁", "photons"), "amp2": ("A₂", "photons"), "amp3": ("A₃", "photons"),
        "offset": ("Offset", "photons"),
        "reduced_chi_square": ("χ²ᵣ", ""),
        "centroid_x": ("Centroid x", "px"), "centroid_y": ("Centroid y", "px"),
        "a2": ("α₂", "%"), "a3": ("α₃", "%"),
    }

    def compose(channel, symbol, unit):
        label = f"{channel} {symbol} ({unit})" if unit else f"{channel} {symbol}"
        if engine == "mpl":
            # Convert Plotly <sub>/<sup> markup to Matplotlib mathtext, e.g.
            # "τ<sub>mod</sub>" -> "$τ_{\\mathrm{mod}}$". Unicode subscripts are untouched.
            label = re.sub(r"(\S)<sub>(.*?)</sub>",
                           lambda m: "$" + m.group(1) + r"_{\mathrm{" + m.group(2) + "}}$", label)
            label = re.sub(r"(\S)<sup>(.*?)</sup>",
                           lambda m: "$" + m.group(1) + r"^{\mathrm{" + m.group(2) + "}}$", label)
        return label

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
