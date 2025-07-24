# FLIM Playground

Input Type, Channels, Extracted Feature Types
- Lifetime 
    - fit
    - fit free (e.g. Phasor)
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
pyinstaller Flim-Playground.spec --clean
```

# Documentation
- @[docs]()


# TODO
- add reference dye option for calibration phasor
- add flimlib
- finish morphology and texture
- data analysis (vis + classification) config
- add confidence interval to effect size 
- move config away?
- fix the color order?

```bash
streamlit run main.py # when in development
```

```bash
python launcher.py # check for building 
```