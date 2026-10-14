"""Export analyses as standalone Matplotlib scripts with editable settings.

Inline shared computation helpers with inspect.getsource() for app/export parity.
String templates supply the Matplotlib rendering and script orchestration. Each
method lives in its own module and registers a MethodSpec below.
"""
from src.export_script import (
    classification,
    dimension_reduction,
    distribution_2d,
    feature_comparison,
    feature_histogram,
    phasor,
)
from src.export_script.feature_comparison import get_effect_size_threshold_capture
from src.export_script.mpl_helpers import scatter_with_encodings
from src.export_script.skeleton import (
    MethodSpec,
    _build_config_section,
    _build_data_loading,
    _build_derived_export_helpers,
    _build_filters,
    _build_footer,
    _build_preamble,
    _build_read_call,
)

__all__ = ["METHODS", "MethodSpec", "_build_read_call", "generate_script",
           "get_effect_size_threshold_capture", "scatter_with_encodings"]

# One entry per method module, keyed by the display name the page sends as state["method"].
METHODS = {spec.name: spec for spec in (
    feature_comparison.SPEC, feature_histogram.SPEC, distribution_2d.SPEC,
    phasor.SPEC, dimension_reduction.SPEC, classification.SPEC)}

# A method with no module still gets a runnable script that says so.
UNSUPPORTED = MethodSpec(
    name="",
    build=lambda state: f'\nprint("Export not yet supported for method: {state["method"]}")\n',
    config_lines=lambda state: [],
    imports=lambda state: [],
    packages=lambda state: set(),
    footer=lambda state: "plt.tight_layout()",
)


def generate_script(state: dict) -> str:
    """Generate a self-contained Python script from the current analysis state."""
    spec = METHODS.get(state["method"], UNSUPPORTED)
    parts = [
        _build_preamble(spec, state),
        _build_config_section(spec, state),
        _build_data_loading(state),
        _build_filters(state),
        _build_derived_export_helpers(spec),
        spec.build(state),
        _build_footer(spec, state),
    ]
    return "\n".join(parts)
