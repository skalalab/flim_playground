# FLIM Playground

<p align="center">
  <img src="logo.png" alt="FLIM Playground Logo">
</p>

FLIM Playground allows you to extract single-cell features from fluorescence lifetime imaging microscopy (FLIM) raw data (**Data Extraction**) and analyze extracted features or your own datasets using a built-in repertoire of visual-analytic modules (**Data Analysis**).

# Data Extraction Demo
- Demo uses the T cell activation [dataset](example_data/Data_Extraction/T_cell_activation) from this [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11425855/):

https://github.com/user-attachments/assets/a01b8a22-1bc3-46f1-aa37-1c3191a6fa1a

# Data Analysis Demo
- Demo uses the inhibitor treatments on cancer cell lines (MCF7 and PANC-1) [dataset](./example_data/Data_Analysis/inhibitors.csv) extracted by Data Extraction:

https://github.com/user-attachments/assets/7ac6b61f-7bde-45b8-92f5-5dbdb05dde67

## Data Analysis on mobile phone

https://github.com/user-attachments/assets/246c13f3-a8ca-4c17-9e2c-0c5cf961e28a

## Use Your Own Data in Data Analysis
- Demo uses the [iris dataset](example_data/Data_Analysis/iris.csv) and the [wine quality dataset](example_data/Data_Analysis/wine_quality.csv):
  
https://github.com/user-attachments/assets/08b55f51-c7a6-4fa3-a00a-65f3fcd11cc6

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

FLIM Playground is currently on [bioarchive](https://www.biorxiv.org/content/10.1101/2025.09.30.679625). If it contributed to your research—whether through Data Extraction for single-cell feature extraction or through Data Analysis for data exploration, visualization, selection of analysis methods, or hyperparameter tuning (UMAP, clustering, classification, etc.)—please cite this work in your publication. Your citation directly supports us in maintaining and improving it ✨🎈🍾. 

# TODO

- [x] randomize the plot order of points so specific colors and shapes plotted later do not occlude early color and shape groups. 
- [x] add multiple analysis config profile.
- [x] tune classification thresholds based on user-specified metrics (to combat class inbalance). 
- [x] add filters based on numerical user-specified feature range/cutoff 
- [x] overlaying box plot to feature comparison
- [x] based on the light/dark mode, the plot title, axis labels, and tick labels need to be adjusted
- [x] add an interactive way in feature comparison to reorder the x-axis groups
- [x] log scale for x-y axis in 2d scatter plot, y-axis in feature comparison and x-axis in feature histogram
- [x] freeze the umap axis limits so that the user can use legend toggles to explore the subgroups in umap
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
