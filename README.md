# FLIM Playground
An interactive, ui-based data-centric playground that focus on FLIM analysis. It includes data/feature extraction from inputs (regionprops), unsupervised clusetering visualization (UMAP, PCA), outlier finder based on extracted features or raw data, classification/prediction (e.g. random forest), a SDT toolbox that has tool such as single-cell phasor plots, and simple plots (e.g. box/swarm plot with t-tests between conditions on a given feature). 

# Installation
Due to the latency/speed constraint, some modules (regionprops, classification) of the playground are better to be run offline. To do that, user needs to have the python environment installed on their local desktop. For other modules (e.g. visualizations), user can go to the [website](https://flim-playground.streamlit.app/) without the need to install *anything*. 

## Environment Setup
Just as we need the `seg`/`segnew` environment to run the `cell-analysis-tools` and segment the cell, we need an environment to run this playground. After downloading this repository from github and navigating to it, you can run `conda env create -f environment.yml` and conda will take care of the setup process. The new environment includes packages from `seg` such as `napari`, `cellpose`, and `cell-analysis-tools`. 

## Run the app offline
To run it, use the command `streamlit run main.py` in anaconda prompt/terminal. The app will spin up webpages to your default web browser. 

# Modules
## RegionProps 
To be developed. 

## Cluster & Outlier finder

### Dimension Reduction 
Flim playground offers two dimension reduction algorithms, Principal Componenet Analysis (PCA) and UMAP, to provide unsupervised clustering on the input data. Then it selects and visualizes the first two dimensions of the reduced data (the output of the dimension reduction algorithm). The visualization can help people identify **clusters** and **outliers**. 

### Outlier finder
Thus, the tool can identify outliers by: 
- performing dimension reduction on selected subset of fitted parameters
    - Fitting Lifetime images using SPCImage can be error-prone, especially when analyzing big datasets on one's own. 
    - Therefore, a sanity check method is in need. We use principal component analysis on your uploaded datasets using selected lifetime and morphological variables. PCA is chosen over UMAP and t-SNE because of its speed, which makes it approiate for online usage. 
- performing dimension reduction on raw inputs (sdt and mask)
    - To be developed. 
    - Some outliers are not caused by errors using SPCImage. Instead, it may be due to inconsistencies occurred during image/data acquisition or masking. Therefore, PCA can be performed on input data (SDTs and Masks) directly: each cell occupies a set of spatial pixels, under each of which is 256 time bins. We can sum the times spatially to get the cell-level 256 time bins and perform PCA on them. It does not need the IRF file. 

### Design 
You can identify the outliers by hovering over the points that show the `base_name` (we assume you dataset has a column called `base_name`, which is of this format: `image_name` + `_` + `cell_number`). You can select the outlier by clicking on it, then you will have the option to remove it and cells that belongs to the same image. Then the algorithm and visualization will be rerun on the new data without the outlier image. The rationale is that if all the outlier points belong to the same image, them that image should probably be reanalyzed. 

For big datasets such as the redox dataset, which has multiple experiments, cell lines, treatments, and media, it offers filtering based on those columns. 

Additionally, it can
- plot image-level boxplots on a selected fitted parameters.

## SDT toolbox
To be developed. 

## Classification
To be developed. 

## Visualizations
To be developed. 

# Deployment 
It is deployed on using streamlit's cummunity server at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). It is free and will automatically fetch the new commits and update the app. The first time visit will have longer delay because Streamlit is managing in the background to set up machines and a python environment to run this app. 

# Progress
- 12/6/24: bootstrapping the app with very minimal elements. Finished the deployment. 
- 12/9/24: working on designing the modules and the layout of the app. 
- 12/11/24 - 12/13/24: working on the clustering & outlier finder module.
- 12/14/24: working on adding filtering mechanism for cluster& outlier finder module to support exploring big datasets. TODO: add more coloring mechanism so that it can color by more than one columns (i.e. combination of columns)