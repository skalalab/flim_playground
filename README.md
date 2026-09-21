# FLIM Playground

<p align="center">
  <img src="logo.png" alt="FLIM Playground logo" width="360">
</p>

<p align="center">
  <a href="https://github.com/skalalab/flim_playground/releases/latest"><img src="https://img.shields.io/github/v/release/skalalab/flim_playground?label=latest%20release" alt="Latest release"></a>
  <a href="https://flim-playground.streamlit.app/"><img src="https://img.shields.io/badge/try-live%20demo-2ea44f" alt="Try the live demo"></a>
  <a href="https://skalalab.github.io/flim_playground_doc/"><img src="https://img.shields.io/badge/docs-online-0969da" alt="Online documentation"></a>
  <a href="https://doi.org/10.5281/zenodo.19744706"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19744706.svg" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/skalalab/flim_playground" alt="MIT license"></a>
</p>

FLIM Playground is a **FLIM-focused framework for multimodal single-cell imaging**. It extracts measurements from fluorescence lifetime imaging microscopy (FLIM) data, supports quantitative phase imaging (QPI) as an additional modality, and analyzes your own tabular datasets through interactive visual-analytic modules. Its shared configuration, feature schema, and export workflow are designed to expand to additional imaging modalities.

**[Try Data Analysis in your browser](https://flim-playground.streamlit.app/)** · **[Download the desktop app](https://github.com/skalalab/flim_playground/releases)** · **[Read the documentation](https://skalalab.github.io/flim_playground_doc/)** · **[Browse example data](./example_data/)**

## What FLIM Playground does

### FLIM data extraction

Use the desktop application to configure channels, calibrate measurements, fit fluorescence lifetimes, and extract single-cell features from FLIM image data. Extracted features can be carried directly into Data Analysis.

### Analyze any tabular dataset

Data Analysis accepts CSV, TSV/TXT, Excel, and OpenDocument tables from any acquisition modality. A column-review step identifies row IDs, categorical metadata, measurements, and feature groups before you explore the data with:

- feature histograms and comparisons;
- two-dimensional distributions, phasor plots, and dimensional reduction;
- filtering, grouping, classification, clustering, and model tuning; and
- publication-ready exports, including a standalone Python script.

The included examples work even when a table was not produced by FLIM Playground.

### Designed for additional modalities

FLIM is the primary workflow today, with quantitative phase image (QPI) and generic two dimensional imaging modalities (e.g.  widefield, dark/brightfield, etc.) already integrated. Shared configuration profiles, feature roles, derived features, and analysis modules provide a common path for future imaging modalities without changing how users explore single-cell measurements.

## Try it in 60 seconds

1. Open the [browser demo](https://flim-playground.streamlit.app/).
2. Go to **Data Analysis** and upload the included [inhibitors.csv](./example_data/Data_Analysis/inhibitors.csv) dataset.
3. Review the detected column roles and feature groups.
4. Open a histogram, feature comparison, or dimensional-reduction module.
5. Export the result or download the Python script that reproduces the analysis.

The hosted demo is intended for analysis of example or non-sensitive data. Use the desktop app for local processing and for FLIM Data Extraction.

## See it in action

- [Watch the Data Extraction demo](https://github.com/user-attachments/assets/8391d0ca-e0a7-49ef-9f8b-3d7e0d9a273f)
- [Watch the Data Analysis demo](https://github.com/user-attachments/assets/f422d364-dff8-4422-afdb-48601bd7e0dd)

## Example datasets

- [Inhibitor treatments](./example_data/Data_Analysis/inhibitors.csv) — single-cell features from MCF7 and PANC-1 experiments.
- [Iris](./example_data/Data_Analysis/iris.csv) — a compact table for trying feature selection and visualization.
- [Wine quality](./example_data/Data_Analysis/wine_quality.csv) — a larger general-purpose tabular analysis example.
- [FLIM extraction examples](./example_data/Data_Extraction/README.md) — T-cell activation data, reference files, and a decay archive.

## Installation

### Browser demo

Use the [live Data Analysis demo](https://flim-playground.streamlit.app/) without installing anything. It demonstrates table-based analysis; FLIM Data Extraction requires the desktop application.

### Option 1: Download from Releases

The desktop application includes Data Extraction and Data Analysis and keeps your input data on your computer. Builds for macOS, Windows 11, and Ubuntu 24.04 LTS are published under [Releases](https://github.com/skalalab/flim_playground/releases).

<details>
<summary><strong>macOS</strong></summary>

Install *and* upgrade with one paste into **Terminal** (find it with Spotlight: ⌘-Space, type "Terminal"): it downloads the build and unpacks **Flim-Playground.app** into your Downloads folder, ready to double-click. Any previous copy is replaced; a download that fails leaves it untouched. Use the block for your Mac (unsure which? **Apple menu → About This Mac**).

**Apple Silicon** (M1 and later):

```bash
curl -fL -o ~/Downloads/Flim-Playground-mac.tar.gz \
  https://github.com/skalalab/flim_playground/releases/latest/download/Flim-Playground-mac.tar.gz &&
  rm -rf ~/Downloads/Flim-Playground.app &&
  tar -xzf ~/Downloads/Flim-Playground-mac.tar.gz -C ~/Downloads
```

**Intel Macs:**

```bash
curl -fL -o ~/Downloads/Flim-Playground-mac-intel.tar.gz \
  https://github.com/skalalab/flim_playground/releases/latest/download/Flim-Playground-mac-intel.tar.gz &&
  rm -rf ~/Downloads/Flim-Playground.app &&
  tar -xzf ~/Downloads/Flim-Playground-mac-intel.tar.gz -C ~/Downloads
```

</details>

<details>
<summary><strong>Windows</strong></summary>

Download `Flim-Playground-Setup.exe`, run the installer, then launch from the **Start Menu** shortcut it creates.

#### First launch: getting past the security warning

This applies to **Windows** only — the macOS `curl` install above never triggers a warning. FLIM Playground is distributed **without a paid code-signing certificate**, so Microsoft Defender SmartScreen flags the installer because it "isn't commonly downloaded" yet. This is expected for open-source apps shipped outside the Microsoft Store — nothing is wrong with the download, and you can always [build from source](#option-2-build-from-source) if you'd rather verify it yourself. You only need to clear the warning **once per download**.

- **In your browser:** if the download is flagged, click **⋯ → Keep**, then **Keep anyway** when it double-checks.

<img src="assets/security-win-1-keep.png" width="380" alt="Browser download menu: Keep"> <img src="assets/security-win-2-smartscreen.png" width="300" alt="SmartScreen: Keep anyway">

- **When you run it:** double-click `Flim-Playground-Setup.exe`; if a blue *"Windows protected your PC"* box appears, click **More info → Run anyway**, then proceed through the installer.

</details>

<details>
<summary><strong>Linux (Ubuntu 24.04+)</strong></summary>

Download `Flim-Playground-linux.tar.gz` and **double-click it to extract** (or right-click → *Extract* in your file manager). You get a single **`Flim-Playground-linux`** folder — open it and run `./install.sh` once to add **FLIM Playground** to your application menu, then click it to launch. (Or run the `Flim-Playground` binary directly.) *Prefer the terminal? Extract with `tar --one-top-level -xzf Flim-Playground-linux.tar.gz` so the files land in their own folder instead of the current directory.*

</details>

#### Upgrading

<details>
<summary>Show upgrade instructions</summary>

Upgrading is the same step as installing — your settings (`config.toml`, plus `analysis_config.toml` if you have one) live **outside** the app, so a new version never touches them: on macOS beside the app, on Windows at the root of the install folder, on Linux in `~/.config/flim-playground/`.

- **macOS** — paste the [same command](#option-1-download-from-releases) again: it fetches the new build, deletes the old app and unpacks the replacement in its place.
- **Windows** — run the new `Flim-Playground-Setup.exe`; it upgrades your existing installation in place. Every download is a fresh unsigned file, so expect the [SmartScreen warning](#first-launch-getting-past-the-security-warning) again each time.
- **Linux** — delete the old `Flim-Playground-linux` folder, extract the new tarball in its place, then re-run `./install.sh` so the menu launcher points at the new files.

</details>

### Option 2: Build from source

Requirements: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/skalalab/flim_playground.git
cd flim_playground
uv sync
uv run streamlit run main.py
```

To build the desktop bundle:

```bash
uv run pyinstaller Flim-Playground.spec --clean
```

## Documentation and support

- [User documentation](https://skalalab.github.io/flim_playground_doc/)
- [Release notes and downloads](https://github.com/skalalab/flim_playground/releases)
- [Report an issue or request a feature](https://github.com/skalalab/flim_playground/issues)
- [Project history](./HISTORY.md)

## Citation

If FLIM Playground contributed to your research, cite both the software version and the companion publication.

**Publication:**

> Zhao, W., Samimi, K., Skala, M.C., and Datta, R. (2026). FLIM Playground: An interactive, end-to-end graphical user interface for analyzing single cells with fluorescence lifetime imaging microscopy. *Cell Reports Methods*. https://doi.org/10.1016/j.crmeth.2026.101484

**Software:**

> Zhao W., Samimi K., Skala M.C., Datta R. *FLIM Playground* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19744706

**Example and validation data:**

> Zhao, W., Samimi K., Skala, M. C., & Datta, R. (2026). *Example and validation datasets for FLIM Playground* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19774943

## Development

```bash
uv run pytest
uv run streamlit run main.py
```

## License

FLIM Playground is released under the [MIT License](LICENSE).

## 🎡 Playground Construction News

- 🔬 **QPI, a new imaging modality** — A channel can now be quantitative phase imaging (QPI): give it an OPD image and a cell mask. FLIM Playground covers configuration (for dry mass conversion), calibration (for background correction), and extraction.
- 🧭 **Separate by, in every module** — Split any plot by a categorical column: stacked rows in *Feature Histogram*, one full-size switchable view in *2D Feature Distribution* and *Phasor Plot*, and an overview beside a highlight grid in *Dimension Reduction*. Statistics, GMM fits, and counts follow the selected category, and the Python export reproduces the whole composition.
- 🫧 **Collapse by** — Pick a categorical column and the single row points collapse to one dot per category within each group, so the plot and its statistics compare category (e.g. patient_id) means in *Feature Comparison* and *2D Feature Distribution*.
- 🧾 **Bring your own table** — Upload CSV, TSV/TXT, Excel, or OpenDocument. A column-review table opens with one row per column, its role (row ID, categorical, measurement) and its feature group already guessed, and saves as an analysis profile — the next file with the same columns picks that profile itself. Row ID and FOV columns are optional.
- ⚡ **Speed & scale** — Point plots switch to WebGL above 5,000 points, so large figures no longer freeze the page on scroll, and lifetime curve fitting runs across CPU cores.
- 🧪 **Derived features** — Build new measurements from arithmetic over existing ones, **including across channels** (redox ratios like `A / (A + B)`), written before extraction ever runs and appended as a **Derived Features** group in *Data Analysis*.
- 🗂️ **Configuration profiles** — Save up to 10 extraction setups (channels, suffixes, extractors, fixed lifetimes, laser rate, …) and switch between them in one click. *Data Analysis* configurations are profile-based too.
- 📜 **Export Data Analysis as a Python script** — Download a standalone, editable Python script that reproduces everything you see, and saves publication-ready SVG.
