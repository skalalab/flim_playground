# FLIM Image Analysis Pipeline
An interactive, ui-based pipeline that takes in FLIM inputs: sdt files, fitted parameter files and masks. The pipeline will include data/feature extraction from inputs (regionprops), outlier finder based on extracted features, unsupervised clusetering visualization (UMAP, PCA), classification/prediction (e.g. random forest), single-cell phasor plots, simple plots (e.g. box/swarm plot with t-tests between conditions on a given feature). 

# Components
## Outlier finder
### Purpose
Fitting Lifetime images using SPCImage can be error-prone, especially when analyzing big datasets on one's own. 
Therefore, a sanity check method is in need. We use principal component analysis on your uploaded datasets and selected lifetime and morphological variables. PCA is chosen over UMAP and t-SNE because of its speed, which makes it approiate for online usage. 

### Design 
You can identify the outliers by hovering over the points that show the `base_name` (we assume you dataset has a column called `base_name` which has the image name and cell number). The rationale is that if all the outlier points belong to the same image, them that image should probably be reanalyzed. 


# Deployment 
It is deployed on using streamlit's cummunity server. It is free and will automatically fetch the new commits and update the app. 

# Progress
- 12/6/24: bootstrapping the app with very minimal elements. Finished the deployment. 
