# FLIM Playground

<p align="center">
  <img src="logo.png" alt="FLIM Playground Logo">
</p>

FLIM Playground allows you to extract single-cell features from fluorescence lifetime imaging microscopy (FLIM) raw data (**Data Extraction**) and analyze extracted features or your own datasets using a built-in repertoire of visual-analytic modules (**Data Analysis**).

# Data Extraction + Data Analysis Demo
- Demo uses the T cell activation [dataset](example_data/Data_Extraction/T_cell_activation) from this [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11425855/):

https://github.com/user-attachments/assets/31910280-ae9e-4db9-a1c7-88c81d8d1e05

# Data Analysis Demo
- Demo uses the inhibitor treatments on cancer cell lines (MCF7 and PANC-1) [dataset](./example_data/Data_Analysis/inhibitors.csv) extracted by Data Extraction:
  
https://github.com/user-attachments/assets/bb72c4e3-5785-4770-83ea-690ed3a3cf79

## Use Your Own Data in Data Analysis
- Demo uses the [iris dataset](example_data/Data_Analysis/iris.csv):
  
https://github.com/user-attachments/assets/b85cef5a-3c42-4b0e-a4fa-77968ad5a1f6

# Quick try 
It is deployed at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). 
You can try out analysis modules in the **Data Analysis** section using this sample [dataset](./example_data/Data_Analysis/inhibitors.csv) extracted previously by the **Data Extraction** module.

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

# Citation

FLIM Playground is currently on [bioarchive](https://www.biorxiv.org/content/10.1101/2025.09.30.679625). If you used Data Extraction to get single cell features, or Data Analysis to explore your data to find data of interest, perform analysis, or pin down suitable hyperparameters (UMAP, classification options, etc.) and analysis methods, please cite us 🥳🎉🥂. 

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
