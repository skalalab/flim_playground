# FLIM Playground

![](logo.gif)

FLIM Playground allows you to extract single-cell features from fluorescence lifetime imaging microscopy (FLIM) raw data (Data Extraction) and analyze extracted features or datasets extracted via other methods using a built-in repertoire of methods (Data Analysis).

# Data Extraction Demo

# Data Analysis Demo

https://github.com/user-attachments/assets/bb72c4e3-5785-4770-83ea-690ed3a3cf79

## Use Your Own Data in Data Analysis

https://github.com/user-attachments/assets/b85cef5a-3c42-4b0e-a4fa-77968ad5a1f6

# Quick try 
It is deployed at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). 
You can try out analysis modules in the **Data Analysis** section using this sample [dataset](./inhibitors.csv) extracted previously by the **Data Extraction** module

# Install
## Option 1: Download from Releases
- Releases for Mac OS 26, Windows 11 (look for the latest ones)
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
- @[docs](https://skalalab.github.io/flim_playground_doc/)

# TODO
- add contour map that use color to encode data point density
- randomize the plot order of points
- add hierarchical clustering
- add linear mixed effect model
- add modality alignment
- add confidence interval to effect size

```bash
streamlit run main.py # when in development
```

```bash
python launcher.py # check for building 
```
