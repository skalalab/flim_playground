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

### QPI as an additional modality

Use a quantitative phase image and cell mask as a channel, configure dry-mass conversion and background correction, and combine QPI measurements with FLIM and other single-cell features.

### Analyze any tabular dataset

Data Analysis accepts CSV, TSV/TXT, Excel, and OpenDocument tables from any acquisition modality. A column-review step identifies row IDs, categorical metadata, measurements, and feature groups before you explore the data with:

- feature histograms and comparisons;
- two-dimensional distributions, phasor plots, and dimensional reduction;
- filtering, grouping, classification, clustering, and model tuning; and
- publication-ready exports, including a standalone Python script.

The included examples work even when a table was not produced by FLIM Playground.

### Designed for additional modalities

FLIM is the primary workflow today, with QPI already integrated as an additional modality. Shared configuration profiles, feature roles, derived features, and analysis modules provide a common path for future imaging modalities without changing how users explore single-cell measurements.

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

### Desktop application

Download the latest build for macOS, Windows 11, or Ubuntu 24.04 LTS from [Releases](https://github.com/skalalab/flim_playground/releases). The desktop application includes Data Extraction and Data Analysis and keeps your input data on your computer.

### Build from source

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
