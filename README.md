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

- 🗺️ **Dimension Reduction separation grid** — Keep a combined UMAP, PCA, or t-SNE overview beside smaller maps highlighting individual groups. **Separate by** accepts two categorical columns: the first defines rows and the second defines columns. Color, shape, and opacity remain independent, and the Python export reproduces the whole composition in one SVG.
- 🎨 **Point encoding in one row** — In *Feature Comparison*, use **Opacity | Subcolor | Shape** to choose how one categorical column decorates the points. Switching modes takes one click and keeps the selected column; clearing the column turns that encoding off. Subcolor assigns consistent colours across groups and changes *Color by* to *Group by*. The four controls are *Separate by*, *Color/Group by*, *Collapse by*, and the shared point encoding. Other point plots keep separate shape and opacity controls. Missing data (`N/A`) stays outside the opacity ramp so it can never outrank a real level.
- ⚡ **Speed & scale** — Point plots switch to WebGL rendering above 5,000 drawn points, so large figures no longer freeze the page on scroll; styling tweaks restyle the existing figure instead of rebuilding it, and the KDE overlay is no longer quadratic in point count. In *Data Extraction*, lifetime curve fitting runs across CPU cores via multiprocessing, and raw-data checks are keyed on the files they actually read, so reassigning one channel no longer re-decodes every later channel's files.
- 🎯 **"Except:" selections** — Every categorical filter and feature picker takes an *Except:* mode, so keeping all-but-a-few is one click instead of many. Filters also narrow symmetrically against each other, so the order of columns in your config no longer decides which filter combinations are reachable.
- 🧪 **Derived feature extraction & analysis** — Build custom mathematical features (e.g., redox ratios like `A / (A + B)`, or ratio / difference formulas) using arithmetic expressions over existing features. These are appended as `Derived: <name>` columns and automatically consolidated into a unified **Derived Features** group in the Data Analysis layer.
- 🗂️ **Multiple configuration profiles** — Save up to 10 named setups in *Data Extraction* (channels, file suffixes, feature extractors, fixed lifetimes, laser rate, …) and switch between them in one click from the Configuration page. *Data Analysis* configurations are profile-based too, so you can keep several datasets' settings side by side.
- 📜 **Export Data Analysis as a Python script** — Download a standalone, editable Python script that reproduces all *Data Analysis* settings you see in FLIM Playground; also saves figures as publication-ready SVG.

# Data Extraction Demo
- Demo uses the T cell activation [dataset](example_data/Data_Extraction/T_cell_activation) from this [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11425855/):

https://github.com/user-attachments/assets/a01b8a22-1bc3-46f1-aa37-1c3191a6fa1a

# Data Analysis Demo
- Demo uses the inhibitor treatments on cancer cell lines (MCF7 and PANC-1) [dataset](./example_data/Data_Analysis/inhibitors.csv) extracted by Data Extraction:

https://github.com/user-attachments/assets/7ac6b61f-7bde-45b8-92f5-5dbdb05dde67

## Compare groups in Dimension Reduction

Choose **Separate by** after selecting your features and reduction method:

| Selection | Display |
| --- | --- |
| None | One combined overview |
| One categorical column | Overview beside one vertical column of small maps, one per value |
| Two categorical columns | Overview beside a matrix, with the first selection defining rows and the second defining columns |

UMAP, PCA, and t-SNE use consistent plotting frames that are wider than they are
tall. Coordinate ranges are padded to preserve equal x/y scales without
stretching the embedding. The chart fits the available screen height.
Every panel uses the same embedding, coordinate ranges, and aspect ratio. The
overview and the complete grid align at their top and bottom edges, including
when resizing the chart or opening it in fullscreen. Zooming or panning one
panel updates the others. Small maps highlight matching observations and show the
remaining observations in gray as context. Black bottom and left axis lines
separate the panels; gridlines remain hidden.
Small-map points are two units smaller than overview points (minimum size 1),
and their row and column labels follow **Legend Font Size**.
The right margin fits the labels, and the legend has its own space below the
overview's x-axis title.
Empty matrix intersections retain their panel. Values follow natural order, and
missing categories appear as `N/A`.

**Color by**, **Shape by**, and **Opacity by** can reuse separation columns. Each
Color by combination retains its own palette color across all panels. The shared
legend controls foreground points throughout the figure. **Show group counts**
reports unique observations in the shared legend. Gray context and repeated
displays do not inflate those counts.

Filtering and removal of incomplete feature observations happen before fitting.
Changing separation or visual encodings reuses the
cached coordinates. Separation settings survive method changes and column review;
a selected separation column remains selected when filtering leaves one value.
The standalone Python export includes these settings and saves the full view as
one SVG. This separation grid is available in Dimension Reduction.

## Use Your Own Data in Data Analysis
- Demo uses the [iris dataset](example_data/Data_Analysis/iris.csv) and the [wine quality dataset](example_data/Data_Analysis/wine_quality.csv):
  
https://github.com/user-attachments/assets/08b55f51-c7a6-4fa3-a00a-65f3fcd11cc6

# Quick try 
It is deployed at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). 
You can try out analysis modules in the **Data Analysis** section using this sample [dataset](./example_data/Data_Analysis/inhibitors.csv) extracted previously by the **Data Extraction** module.

# Install
## Option 1: Download from Releases
Grab the latest build for your OS under the **Releases** tab on the right (available for macOS, Windows 11, and Ubuntu 24.04 LTS):
- **macOS** — download `Flim-Playground-mac.tar.gz` (Apple Silicon: M1 and later) or `Flim-Playground-mac-intel.tar.gz` (Intel Macs — check **Apple menu → About This Mac** if unsure), unzip, run the [one-line Terminal command below](#first-launch-getting-past-the-security-warning) once, then double-click **Flim-Playground.app**.
- **Windows** — download `Flim-Playground-Setup.exe`, run the installer, then launch from the **Start Menu** shortcut it creates.
- **Linux** (Ubuntu 24.04+) — download `Flim-Playground-linux.tar.gz` and **double-click it to extract** (or right-click → *Extract* in your file manager). You get a single **`Flim-Playground-linux`** folder — open it and run `./install.sh` once to add **FLIM Playground** to your application menu, then click it to launch. (Or run the `Flim-Playground` binary directly.) *Prefer the terminal? Extract with `tar --one-top-level -xzf Flim-Playground-linux.tar.gz` so the files land in their own folder instead of the current directory.*

### First launch: getting past the security warning
FLIM Playground is distributed **without a paid code-signing certificate**, so the first time you open a downloaded build, Windows and macOS show a security warning. This is expected for open-source apps shipped outside the App Store / Microsoft Store — nothing is wrong with the download, and you can always [build from source](#option-2-build-from-source) if you'd rather verify it yourself. You only need to clear the warning **once per download**.

**Windows** — Microsoft Defender SmartScreen flags the installer because it "isn't commonly downloaded" yet:

- **In your browser:** if the download is flagged, click **⋯ → Keep**, then **Keep anyway** when it double-checks.

  <img src="assets/security-win-1-keep.png" width="380" alt="Browser download menu: Keep"> <img src="assets/security-win-2-smartscreen.png" width="300" alt="SmartScreen: Keep anyway">

- **When you run it:** double-click `Flim-Playground-Setup.exe`; if a blue *"Windows protected your PC"* box appears, click **More info → Run anyway**, then proceed through the installer.

**macOS** — the app isn't notarized by Apple, so **before double-clicking it for the first time**, open **Terminal** (find it with Spotlight: ⌘-Space, type "Terminal"), paste this one line, and press Return:

```bash
xattr -dr com.apple.quarantine ~/Downloads/Flim-Playground.app
```

This strips the download-quarantine flag, so the app opens with a normal double-click and you skip macOS's security pop-ups entirely. If the app is somewhere other than Downloads (e.g. `/Applications`), point the command at that path instead.

**Prefer a single step?** macOS only sets the quarantine flag on files downloaded by a *browser* — `curl` does not set it at all. Download that way and there is nothing to strip and no warning to clear. Instead of downloading from the Releases tab, paste the two lines for your Mac into Terminal (unsure which? **Apple menu → About This Mac**):

**Apple Silicon** (M1 and later):

```bash
curl -fL -o ~/Downloads/Flim-Playground-mac.tar.gz \
  https://github.com/skalalab/flim_playground/releases/latest/download/Flim-Playground-mac.tar.gz &&
  tar -xzf ~/Downloads/Flim-Playground-mac.tar.gz -C ~/Downloads
```

**Intel Macs:**

```bash
curl -fL -o ~/Downloads/Flim-Playground-mac-intel.tar.gz \
  https://github.com/skalalab/flim_playground/releases/latest/download/Flim-Playground-mac-intel.tar.gz &&
  tar -xzf ~/Downloads/Flim-Playground-mac-intel.tar.gz -C ~/Downloads
```

Either way you get **Flim-Playground.app** in your Downloads folder — just double-click it. No `xattr`, no pop-ups.

If you double-clicked first and got blocked with one of the errors below — no problem: run the same command, then double-click again.

<img src="assets/security-mac-1-blocked.png" width="270" alt="macOS: Apple could not verify"> <img src="assets/security-mac-2-error-47.png" width="270" alt="macOS: error -47">

### Upgrading
Already running an older version? Grab the latest build from the **Releases** tab, then follow the steps for your platform below. Because every download is a fresh, unsigned file, the [security warning](#first-launch-getting-past-the-security-warning) above **reappears for each new version** — clear it the same way each time (on macOS, re-run the `xattr` command on the new download — or use the `curl` upgrade command below, which never triggers the warning in the first place). Your settings — `config.toml` (Data Extraction) and `analysis_config.toml` (Data Analysis, if you have one) — are **not** bundled inside the app, so they carry over. Where they are stored, and what that means when you upgrade, differs by platform:

- **macOS** — the whole upgrade is one paste into **Terminal**, which downloads the new build, removes the old app and unpacks the new one in its place. The three steps are chained (`&&`) and `curl` is asked to fail on an HTTP error (`-f`), so a download that does not arrive leaves the app you already have untouched. Use the block for your Mac (unsure which? **Apple menu → About This Mac**); if you keep the app somewhere other than Downloads, adjust all three paths.

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

  Because `curl` never sets the download-quarantine flag, **there is no `xattr` step and no security warning** — just double-click the new app. Prefer the browser? Download the tarball from the Releases tab as before, delete the old **Flim-Playground.app**, extract the new one in its place, then re-run the [`xattr` command](#first-launch-getting-past-the-security-warning) on it. Either way your settings are saved in the folder *beside* the app, **outside** the `.app` bundle, so swapping the app never touches them — just keep the two `.toml` files where they are.
- **Windows** — run the new `Flim-Playground-Setup.exe`; it upgrades your existing installation in place. Your settings sit at the **root of the install folder** (next to the program, not inside the internal payload the installer refreshes), so they are preserved automatically.
- **Linux** — your settings are saved in `~/.config/flim-playground/` (following the XDG convention), **outside** the app folder, so replacing the app never touches them. Delete the old `Flim-Playground-linux` folder, double-click the new tarball to extract it in the same place, then re-run `./install.sh` so the menu launcher points at the new files — your settings are picked up automatically.

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
