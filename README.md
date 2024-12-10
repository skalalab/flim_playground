# FLIM Image Analysis Pipeline
An interactive, ui-based pipeline that takes in FLIM inputs: sdt files, fitted parameter files and masks. The pipeline will include data/feature extraction from inputs (regionprops), outlier finder based on extracted features or input data itself (sdt), unsupervised clusetering visualization (UMAP, PCA), classification/prediction (e.g. random forest), single-cell phasor plots, simple plots (e.g. box/swarm plot with t-tests between conditions on a given feature). 

# Installation
Due to the latency/speed constraint, some steps (regionprops, classification) of the pipeline are better to be run offline. To do that, user needs to have the python environment installed on their local desktop. For other steps (visualization), user can go to the [website](#Deployment) and just upload their data sheet or specify the path to the inputs and run the steps online without the need to install *anything*. 

# Components
## Outlier finder
There are two ways to identify outliers: perform dimension reduction on
1. fitted parameters
2. inputs (sdt and mask)
### On fitted parameters 
#### Purpose
Fitting Lifetime images using SPCImage can be error-prone, especially when analyzing big datasets on one's own. 
Therefore, a sanity check method is in need. We use principal component analysis on your uploaded datasets using selected lifetime and morphological variables. PCA is chosen over UMAP and t-SNE because of its speed, which makes it approiate for online usage. 

### On inputs directly
#### Purpose
Some outliers are not caused by errors using SPCImage. Instead, it may be due to inconsistencies occurred during image/data acquisition or masking. Therefore, PCA can be performed on input data (SDTs and Masks) directly: each cell occupies a set of spatial pixels, under each of which is 256 time bins. We can sum the times spatially to get the cell-level 256 time bins and perform PCA on them. 

### Design 
You can identify the outliers by hovering over the points that show the `base_name` (we assume you dataset has a column called `base_name` which has the image name and cell number). The rationale is that if all the outlier points belong to the same image, them that image should probably be reanalyzed. 


# Deployment 
It is deployed on using streamlit's cummunity server. It is free and will automatically fetch the new commits and update the app. 

# Progress
- 12/6/24: bootstrapping the app with very minimal elements. Finished the deployment. 
