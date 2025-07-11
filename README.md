# FLIM Playground

Input Type, Channels, Extracted Feature Types
- Lifetime 
    - Fit
    - Fit free (e.g. Phasor)
- Intensity
    - morphology
    - texture


# Quick try 
It is deployed at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). 
You can try out the **Visualization** and **Classification** modules using this sample [dataset] extracted previously by the **Data Extraction** module

# Install
## Option 1: Download from Releases
- Releases for Mac OS 15, Windows 11
## Option 2: Build from source
### Clone the repo
Navigate into the repository once cloned. 
### Install the python environment
- Install `uv` if not yet installed
- run `uv sync`
### Build
```bash
pyinstaller launcher.py --name "Flim-Playground" --icon logo.png --add-data "src:src" --add-data "pages:pages" --add-data "main.py:." --add-data "config.toml:." --add-data "logo.png:." --hidden-import pages.classification --hidden-import pages.visualization --hidden-import pages.data_extraction --collect-all streamlit --collect-all streamlit_plotly_events --onefile --noconfirm --clean --noconsole
```
- You can see debug output when the app starts by removing the `--noconsole` flag.

# Documentation
- @[docs]()


# TODO
- more rigorous classification 