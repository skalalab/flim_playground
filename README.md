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
You can try out analysis modules in the **Data Analysis** section using this sample [dataset] extracted previously by the **Data Extraction** module

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

- fit validation (sensitivity with ground truth of 2 component solution, Alek)
- add flimlib 
- move config away (while waiting)
- add modality alignment (later)
- add confidence interval to effect size (later)

```bash
streamlit run main.py # when in development
```

```bash
python launcher.py # check for building 
```