# App ↔ exported-script parity harness

Checks that a Data Analysis plot drawn **in the app** and the same analysis re-run from
the **exported Python script** produce the same numbers, using the real example datasets
rather than synthetic fixtures.

`tests/test_export_script_capture.py` already covers this with small synthetic frames and
runs in the normal pytest suite. These harnesses complement it: they are slower (14,101
cells, real UMAP), need `example_data/`, and are meant to be run **on demand** when
touching `src/export_script.py` or anything in `src/vis/`.

They are deliberately named `parity_*.py` / `harness_*.py` so pytest does not collect them.
These sources are tracked; everything they write under `_work/` is gitignored.

## Running

```bash
uv run python tests/parity/run_all.py           # everything, one verdict
uv run python tests/parity/parity_phasor.py     # phasor only
uv run python tests/parity/parity_methods.py    # all other plot methods
uv run python tests/parity/parity_methods.py 2d # one method: hist|gmm|fc|2d|dr|prune
uv run python tests/parity/parity_classify.py   # classification pipeline
uv run python tests/parity/parity_controls.py   # every control, each option
uv run python tests/parity/parity_controls.py 2d  # one section: shared|filters|enc|hist|fc|subcolor|2d|phasor|dr|clf
```

Generated scripts and their outputs land in `tests/parity/_work/` — safe to delete.

Streamlit prints `missing ScriptRunContext!` warnings throughout; that is expected when
calling app functions outside `streamlit run`. To read just the verdicts:

```bash
uv run python tests/parity/run_all.py 2>&1 | grep -E "^\s+\[|^===|passed|FAILED"
```

## How it works

Each check runs both sides and compares:

1. **App side** — `load_app_df()` replays the app's load path (`check_and_fix_df` →
   `coerce_majority_numeric_cols` → the `get_features()` column prune), then calls the
   plot function in `src/vis/` directly and reads numbers back off the Plotly figure.
2. **Export side** — `base_state()` builds the same dict that
   `pages/data_analysis.py::_export_script_button` collects, feeds it to
   `generate_script()`, and executes the result with `runpy`, reading numbers off the
   Matplotlib artists and any derived CSV.

Two things make this work headlessly:

- **`harness_widgets.patch_streamlit()`** — several plot functions render their own
  widgets mid-plot (GMM hyperparameters, the histogram bin width,
  the 2D marginal type). Outside `streamlit run` those return `None`, which would make the
  app path diverge for reasons unrelated to parity. The patch returns each widget's *own*
  default (`value=` / `options[index]`), which is the state the app is in right after
  first render — exactly what the export's state capture assumes when the matching
  session-state key is absent. Pass `overrides` keyed by the widget **label** (matching the
  app source exactly, e.g. `'Marginal Plot Type'`) to flip one on.
- **`enable_derived()`** — flips `SAVE_DERIVED_DATA = False` in the generated script so it
  writes the CSV that the app offers through a download button, giving a per-cell
  comparison (`GMM_group`, `2D_GMM_group`) instead of just pixels.

## What is covered

| Harness | Checks |
|---|---|
| `parity_phasor` | harmonic-scaled lifetime markers, frequency annotation, title — for harmonics 1 and 2 |
| `parity_methods` | histogram bin edges + per-group counts; `GMM_group` per cell on both the intersection and hard-assignment paths; feature-comparison group counts/y-values/jitter (uncollapsed and collapsed), x positions, tick labels, `separate_by` section headers and dividers, effect-size defaults and title; 2D scatter and marginals for all three marginal types; PCA and UMAP embeddings; the `ANALYSIS_COLUMNS` prune |
| `parity_classify` | identical splits, predictions and metrics for Random Forest and threshold-tuned Logistic Regression |
| `parity_controls` | every control on the page, each option in turn — see below |

`parity_methods` proves each METHOD agrees on full data at default settings.
`parity_controls` walks the CONTROL surface instead, on a filtered subset so the matrix
stays quick:

- **Shared** — `color_by` (1 and 2 columns), `opacity_by`, `shape_by`, both together,
  `separate_by`, categorical filters, numerical filters, both together, point size, axis
  label size, legend size, colormap, show-group-counts
- **Filters** (own section, `parity_controls.py filters`) — driven through the app's real
  `filters_widget()` and the page's real `_collect_*` helpers rather than a mirror, so a
  disagreement between filtering, capture and replay actually shows up: no filter ("All"),
  one categorical value, several values, two categorical columns, `>`, `<=`, two chained
  numerical filters, categorical + numerical, two of each, and an out-of-range threshold
  that the widget clamps. Also the `"Except:"` mode (`src/widgets/multiselect_modes.py`),
  which is stored as the sentinel plus the values to drop but captured as the values it
  *keeps*: one exclusion, several, combined with a second categorical filter and with a
  numerical one, plus excluding a whole column (capture only — the app draws nothing, so
  there is no export button to press, but the capture still has to say `isin([])` rather
  than drop the filter). The app masks with `resolve_selections`' narrowed list while the
  export gets the full-frame complement; those agree only because every mask is applied
  conjunctively, which is what the two-filter cases pin
- **Encoding column types** (own section, `parity_controls.py enc`) — every categorical
  column in the example data is a plain string with no missing values, so this section
  runs Feature Comparison against a derived CSV (`_work/controls/encodings.csv`, built
  once) carrying the two kinds that are not: `passage`, whose numeric-looking values sort
  differently numerically than lexically (2, 4, 10), and `batch`, which has real NaNs.
  Both stay in agreement only because each side calls the same `check_and_fix_df`
  (`src/dataset_io.py:294` fills NaN with `"N/A"`, then casts to `str`) — the app keeps
  raw dtypes as `create_shape_groups_and_map` keys while the export does
  `.unique().astype(str)`, so if either side stopped normalising, a numeric column would
  order differently and a missing one would split into `"nan"` on one side and `"N/A"` on
  the other. Checked on shape, on opacity, and on both at once
- **Feature Histogram** — `log_x`, bin width, GMM on/off, intersection threshold, GMM max
  components, GMM min weight
- **Feature Comparison** — `log_y`, boxplot, connect means, both effect sizes ×
  mean/median, both t-tests, selected pairs, custom order, shape/opacity, all at once
- **Collapse by** — one point per replicate, per x group, holding the MEAN of its
  cells. The frame the plot sees is a different SHAPE from the filtered one, so every
  downstream number moves at once: point count, box quartiles, effect-size n. Cases:
  `dish` alone, with `separate_by`, with `subcolor_by` on the same column (the
  SuperPlot — one colour per replicate held across every x group), with `log_y`, with
  boxplot + effect size, and with a decoration FINER than the replicate
  (`shape_by=image_name`), which the collapse drops on both sides. **Set it on both
  halves of `run_fc`** — the plot-function call AND `method_params` — or the harness
  compares a collapsed export against an uncollapsed app and reports a point-count
  mismatch that reads like a jitter bug. The collapse runs after the `notna` drop and
  before the encoding block; that order is what makes `n` count the contributing
  cells and the colour map describe replicates
- **Subcolor** (own section, `parity_controls.py subcolor`) — the channel that takes
  colour away from the x-axis group and gives it to a nested value, Feature Comparison
  only. What needs pinning is what is global rather than per-group: one colour and one
  legend entry per distinct value across the whole figure, its count over the whole
  figure, and the same colour on both sides even though the export generates its own
  palette from the same seed (`src/vis/subcolor_palette.py`, inlined whole by
  `_extract_module_source`). Cases: `dish` under the subset, with `separate_by`, with
  `opacity_by`, with a custom order, five values nested in two colour groups on the full
  frame, counts on, and a NaN-bearing column folded to one `"N/A"` value. `shape_by` is
  absent on purpose — it shares the picker with subcolor, so the two cannot both be on
- **2D** — `log_x`, `log_y`, all three marginal types, regression line, 2D GMM, its two
  hyperparameters, all at once
- **Phasor** — harmonics 1 and 2, laser rate, combined with shape
- **Dimension Reduction** — PCA, UMAP (default and tuned), t-SNE (default and tuned),
  shape+opacity
- **Classification** — all four classifiers, train split, under/oversampling, class
  weight, both threshold-tuning methods, classifier hyperparameters

Each case checks the plotted values, the point cloud (positions included), axis-label and
legend font sizes, and — where relevant — group colours, plus per-method extras (phasor
marker geometry, DR axis labels, classifier metrics and predictions). Any case with
`opacity_by` also compares per-point alpha as a multiset: the engines take opacity in
differently (a scalar `marker.opacity` per Plotly trace against one per-point alpha array
in Matplotlib), so nothing else would notice the export applying a different ramp.

## Known gaps

None open. A `[KNOWN GAP]` line is a difference that is understood but not yet fixed: it
is reported and does not fail the run. If one starts passing it prints `[FIXED]` — the cue
to drop the `known_gap=True` flag so it becomes a real guard. `Results.check()` still
supports the flag; there is currently nothing carrying it.

Resolved this way already:

- "Show group counts (n) in legend" never reached the export — all seven app plot paths
  read `st.session_state["plot_show_group_counts"]` and label through
  `format_group_label()`, but the flag was never captured into the state dict. Now
  captured by `base_state()`, and `format_group_label()` takes an `engine` argument so the
  one helper writes both renderings: `<br>` plus a 0.75em span for Plotly, a plain newline
  for Matplotlib, which renders no markup in legend entries. That difference is also why
  the check needs `_plain()` — comparing the raw legend strings could never match even
  with the export correct, which is what made this look permanent.

- Feature Comparison `separate_by` section spacing — the export advanced a whole slot
  between sections where the app uses `section_spacing = 0.5`, so every later section and
  its divider drifted right. Now covered by the "absolute x positions", "section headers"
  and "section dividers" checks.
- Feature Comparison sina jitter with `shape_by` / `opacity_by` — the export looped over
  colour groups only, so the KDE was fit over a different support and `rng(42)` drew a
  different sequence than the app's per-`(color, shape, opacity, separate)` subgroup loop.
  Every point landed at a different x (mean |dx| ≈ 0.15 of a 1.0 tick spacing) while y and
  group membership matched, which is why only the "point cloud" check caught it. The
  export now splits by encoding subgroup the way `get_point_visual_mappings()` does. Its
  `len(y_data) < 2` branch went with it: that was never crash protection
  (`_estimate_density_1d` already falls back below 2 points) and it also drew small groups
  at `base_alpha=1.0` against the app's 0.7. Now a normal check on the `shape_by`,
  `opacity_by` and `everything` cases.

## Gotchas found the hard way

- The app's 2D marginal selector offers exactly `gaussian fit` | `boxplot` | `violin`.
  Any other string makes the export draw no marginal and look like a parity bug.
- Matplotlib's `violinplot` emits ~5 artists per violin; only the `PolyCollection` bodies
  correspond to the app's traces.
- Significance brackets are vertical line shapes in `fig.layout.shapes`, so a
  `separate_by` divider cannot be found by `x0 == x1` alone; match on the dash style.
- `add_point_legend_traces()` adds shape/opacity legend entries as `x=[None], y=[None]`
  marker traces. The export draws those as empty scatters, so counting them makes the app
  look like it has one extra point per encoding level — `app_point_traces()` drops them.
- Phasor reference geometry (11 lifetime markers) is drawn as marker
  traces in Plotly but `ax.plot` Line2Ds in Matplotlib, so it never reaches
  `scatter_points()`; exclude those trace names before comparing clouds.
- Plotly reports colours as `rgba(r, g, b, a)` strings, which `matplotlib.colors.to_hex`
  cannot parse.
- Controls read from a widget inside the plot function (bin width, GMM hyperparameters,
  marginal type, log toggles) must be set on BOTH sides — pass them to
  `patch_streamlit()` as well as into `method_params`, or the app quietly uses its default
  while the export uses your value.
- Feature Comparison has no x-axis label (the x axis is categorical), so a font-size
  assertion has to skip unlabelled axes.
- The numerical-filter Operator selectbox offers exactly `>` and `<=`. `apply_filters()`
  raises on anything else on purpose — accepting `>=` or `<` would only let a harness test
  a combination the UI cannot produce, and pass.
- `filters_widget()` clamps a threshold to the current filtered frame's min/max and writes
  the clamped value back to session state, so the captured filter is the clamped one, not
  what was typed. Assert against the widget's settled state, not your seed value.
- `patch_streamlit()` resolves a widget as override → `st.session_state[key]` → its own
  default. That middle step is what lets the harness drive real app widgets such as
  `filters_widget()` by seeding the same session keys the app writes.
- The page's `_collect_*` helpers are lifted out of `pages/data_analysis.py` by AST
  (`page_collectors()`); importing the module would execute the whole Streamlit page. The
  page's module-level imports are compiled in alongside them, so a collector that starts
  calling a newly imported helper still runs here — `_collect_categorical_filters` reaching
  for `selection_key` / `chosen_items` used to `NameError` and take the whole filters
  section down, which reads as a parity failure but is only harness staleness.
- `go.Box` / `go.Violin` traces have no `.mode` attribute; use
  `harness_common.app_point_traces()` rather than touching `t.mode` directly.
- Titles differ by rendering engine on purpose: compare against
  `format_feature_label(var, engine='mpl')`, not the Plotly label.
- Keep `base_state()` in sync with `_export_script_button`. If that function gains a key
  and this one does not, the harness silently stops exercising it.
