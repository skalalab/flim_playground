# FLIM Playground

<p align="center">
  <img src="logo.png" alt="FLIM Playground Logo">
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19744706"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19744706.svg" alt="DOI"></a>
</p>

FLIM Playground allows you to extract single-cell features from <span title="can be readily extended to other imaging modalities">fluorescence lifetime imaging microscopy (FLIM)</span>[^1] raw data (**Data Extraction**) and analyze extracted features or your own datasets using a built-in repertoire of visual-analytic modules (**Data Analysis**).

[^1]: can be readily extended to other imaging modalities

## 🎡 Playground Construction News

- 🧭 **Separate by, in every module** — Split any plot by a categorical column: stacked rows in *Feature Histogram*, one full-size switchable view in *2D Feature Distribution* and *Phasor Plot*, and an overview beside a highlight grid in *Dimension Reduction*. Statistics, GMM fits, and counts follow the selected category, and the Python export reproduces the whole composition.
- 🫧 **Collapse by** — Pick a categorical column and the single row points collapse to one dot per category within each group, so the plot and its statistics compare category (e.g. patient_id) means in *Feature Comparison* and *2D Feature Distribution*.
- 🧾 **Bring your own table** — Upload CSV, TSV/TXT, Excel, or OpenDocument. A column-review table opens with one row per column, its role (row ID, categorical, measurement) and its feature group already guessed, and saves as an analysis profile — the next file with the same columns picks that profile itself. Row ID and FOV columns are optional.
- ⚡ **Speed & scale** — Point plots switch to WebGL above 5,000 points, so large figures no longer freeze the page on scroll, and lifetime curve fitting runs across CPU cores.
- 🧪 **Derived features** — Build new measurements from arithmetic over existing ones, **including across channels** (redox ratios like `A / (A + B)`), written before extraction ever runs and appended as a **Derived Features** group in *Data Analysis*.
- 🗂️ **Configuration profiles** — Save up to 10 extraction setups (channels, suffixes, extractors, fixed lifetimes, laser rate, …) and switch between them in one click. *Data Analysis* configurations are profile-based too.
- 📜 **Export Data Analysis as a Python script** — Download a standalone, editable Python script that reproduces everything you see, and saves publication-ready SVG.

# Data Extraction Demo

For an image-based extraction walkthrough, download the [Mosaic NADH example](example_data/Data_Extraction/README.md#mosaic-nadh-images): 25 fields of view with raw decays, cell masks, and a shared IRF.

- Demo uses the T cell activation [dataset](example_data/Data_Extraction/T_cell_activation) from this [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11425855/):

https://github.com/user-attachments/assets/a01b8a22-1bc3-46f1-aa37-1c3191a6fa1a

# Data Analysis Demo
- Demo uses the inhibitor treatments on cancer cell lines (MCF7 and PANC-1) [dataset](./example_data/Data_Analysis/inhibitors.csv) extracted by Data Extraction:

https://github.com/user-attachments/assets/7ac6b61f-7bde-45b8-92f5-5dbdb05dde67

## Use Your Own Data in Data Analysis
- [A walkthrough using a simple dataset](https://skalalab.github.io/flim_playground_doc/data_analysis_config.html)

# Quick try 
It is deployed at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). 
You can try out analysis modules in the **Data Analysis** section using this sample [dataset](./example_data/Data_Analysis/inhibitors.csv) extracted previously by the **Data Extraction** module.

# Install
## Option 1: Download from Releases
Builds for macOS, Windows 11, and Ubuntu 24.04 LTS are published under [**Releases**](https://github.com/skalalab/flim_playground/releases):
- **macOS** — install *and* upgrade with one paste into **Terminal** (find it with Spotlight: ⌘-Space, type "Terminal"): it downloads the build and unpacks **Flim-Playground.app** into your Downloads folder, ready to double-click. Any previous copy is replaced; a download that fails leaves it untouched. Use the block for your Mac (unsure which? **Apple menu → About This Mac**).

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
- **Windows** — download `Flim-Playground-Setup.exe`, run the installer, then launch from the **Start Menu** shortcut it creates.
- **Linux** (Ubuntu 24.04+) — download `Flim-Playground-linux.tar.gz` and **double-click it to extract** (or right-click → *Extract* in your file manager). You get a single **`Flim-Playground-linux`** folder — open it and run `./install.sh` once to add **FLIM Playground** to your application menu, then click it to launch. (Or run the `Flim-Playground` binary directly.) *Prefer the terminal? Extract with `tar --one-top-level -xzf Flim-Playground-linux.tar.gz` so the files land in their own folder instead of the current directory.*

### First launch: getting past the security warning
This applies to **Windows** only — the macOS `curl` install above never triggers a warning. FLIM Playground is distributed **without a paid code-signing certificate**, so Microsoft Defender SmartScreen flags the installer because it "isn't commonly downloaded" yet. This is expected for open-source apps shipped outside the Microsoft Store — nothing is wrong with the download, and you can always [build from source](#option-2-build-from-source) if you'd rather verify it yourself. You only need to clear the warning **once per download**.

- **In your browser:** if the download is flagged, click **⋯ → Keep**, then **Keep anyway** when it double-checks.

  <img src="assets/security-win-1-keep.png" width="380" alt="Browser download menu: Keep"> <img src="assets/security-win-2-smartscreen.png" width="300" alt="SmartScreen: Keep anyway">

- **When you run it:** double-click `Flim-Playground-Setup.exe`; if a blue *"Windows protected your PC"* box appears, click **More info → Run anyway**, then proceed through the installer.

### Upgrading
Upgrading is the same step as installing — your settings (`config.toml`, plus `analysis_config.toml` if you have one) live **outside** the app, so a new version never touches them: on macOS beside the app, on Windows at the root of the install folder, on Linux in `~/.config/flim-playground/`.

- **macOS** — paste the [same command](#option-1-download-from-releases) again: it fetches the new build, deletes the old app and unpacks the replacement in its place.
- **Windows** — run the new `Flim-Playground-Setup.exe`; it upgrades your existing installation in place. Every download is a fresh unsigned file, so expect the [SmartScreen warning](#first-launch-getting-past-the-security-warning) again each time.
- **Linux** — delete the old `Flim-Playground-linux` folder, extract the new tarball in its place, then re-run `./install.sh` so the menu launcher points at the new files.

## Option 2: Build from source
### Clone the repo
```bash
git clone https://github.com/skalalab/flim_playground.git
```
Then Navigate into the repository once cloned. 

### Install the python environment
- Install `uv` if not yet installed
- run `uv sync`
- then run `source .venv/bin/activate` to activate the virtual environment (works in Mac OS, Linux distributions, and Windows Git bash)

### Build
```bash
pyinstaller Flim-Playground.spec --clean
```
This produces a ready-to-run app folder (`dist/Flim-Playground/`, or `Flim-Playground.app` on macOS) you can launch directly. The Windows `Setup.exe` installer is built separately by CI (Inno Setup), so a from-source build on Windows gives you the runnable app folder rather than an installer.

# Documentation
- @[docs](https://skalalab.github.io/flim_playground_doc/)

# Citation

If FLIM Playground contributed to your research — whether through **Data Extraction** for single-cell feature extraction or through **Data Analysis** for data exploration, visualization, selection of analysis methods, or hyperparameter tuning (UMAP, clustering, classification, etc.) — please cite **both** the software version you used and the preprint. Your citation directly supports us in maintaining and improving it ✨🎈🍾.

**Publication:**
> Zhao, W., Samimi, K., Skala, M.C., and Datta, R. (2026). FLIM Playground: An interactive, end-to-end graphical user interface for analyzing single cells with fluorescence lifetime imaging microscopy. Cell Rep. Methods. https://doi.org/10.1016/j.crmeth.2026.101484

**Software: the latest version**
> Zhao W., Samimi K., Skala M.C., Datta R. *FLIM Playground* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19744706

**Raw data used in the paper:**
> Zhao, W., Samimi, K., Skala, M. C., & Datta, R. (2026). *Example and validation datasets for FLIM Playground* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19774943

# TODO

- [] add hierarchical clustering
- [] add linear mixed effect model
- [] add modality alignment
- [] phasor draw gates to filter
- [] add confidence interval to effect size

# Useful Commands
```bash
streamlit run main.py # when in development
```

```bash
python launcher.py # check for building 
```
