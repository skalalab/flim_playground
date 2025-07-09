# FLIM Playground
An interactive, ui-based data-centric playground that focus on FLIM analysis. It includes data/feature extraction from inputs (regionprops), unsupervised clusetering visualization (UMAP, PCA), outlier finder based on extracted features or raw data, classification/prediction (e.g. random forest), a SDT toolbox that has tool such as single-cell phasor plots, and simple plots (e.g. box/swarm plot with t-tests between conditions on a given feature). For tools that are not feasible but interesting and useful that I am aware of, there is also a page for exteral resources. 

# Installation
Due to the latency/speed and the **data security** constraint, some modules (regionprops, classification) of the playground are better to be run offline. To do that, user needs to have the python environment installed on their local desktop. For other modules (e.g. visualizations), user can go to the [website](https://flim-playground.streamlit.app/) without the need to install *anything*. 

## Environment Setup
Just as we need the `seg`/`segnew` environment to run the `cell-analysis-tools` and segment the cells, we need to setup a python environment to run this playground. After downloading this repository from github and navigating to it,  
- for *Windows* users you can run `conda env create -f conda_environment_windows.yml` and conda will take care of the setup process. 
- for *Mac* users you can run `conda env create -f conda_environment_mac.yml`.

The new environment includes packages from `seg` such as `napari`, `cellpose`, and `cell-analysis-tools`, and works with spyder 6. After creating the environment, you may want to run `conda activate flim`. 

## Run the app offline
To run it, navigate to the downloaded repository in anaconda prompt/terminal and use the command `streamlit run main.py`. The app will spin up webpages to your default web browser. 

# Modules
## RegionProps 
To be developed. 

## Cluster & Outlier finder

### Dimension Reduction 
Flim playground offers two dimension reduction algorithms, Principal Componenet Analysis (PCA) and UMAP, to provide unsupervised clustering on the input data. Then it selects and visualizes the first two dimensions of the reduced data (the output of the dimension reduction algorithm). The visualization can help people identify **clusters** and **outliers**. 

### Outlier finder
Thus, the tool can perform clustering and identify outliers by performing dimension reduction: 
- on selected subset of fitted parameters
    - Fitting Lifetime images using SPCImage can be error-prone, especially when analyzing big datasets on one's own. 
    - Therefore, a sanity check method is in need. We use principal component analysis on your uploaded datasets using selected lifetime and morphological variables. PCA is chosen over UMAP and t-SNE because of its speed, which makes it approiate for online usage. 
- on raw inputs (sdt and mask)
    - Some outliers are not caused by errors using SPCImage. Instead, it may be due to inconsistencies occurred during image/data acquisition or masking. Therefore, PCA can be performed on input data (SDTs and Masks) directly: each cell occupies a set of spatial pixels, under each of which is 256 time bins. We can sum the times spatially to get the cell-level 256 time bins and perform PCA on them. It does not need the IRF file. 
    - Copy and paste the path to the folder that contains all your sdt files and masks. This only works in the *offline* mode, as the online app does have access to your local file system. The folder can and is recommended to have sub-folders, each of which contains files for a different treatment. 


### Design 
You can identify the outliers by hovering over the points that show the `base_name` (we assume you dataset has a column called `base_name`, which is of this format: `image_name` + `_` + `cell_number`). You can select the outlier by clicking on it, then you will have the option to remove it and cells that belongs to the same image. Then the algorithm and visualization will be rerun on the new data without the outlier image. The rationale is that if all the outlier points belong to the same image, them that image should probably be reanalyzed. 

For big datasets such as the redox dataset, which has multiple experiments, cell lines, treatments, and media, it offers filtering based on those columns. The color_by option enables user to compare between different treatments, or different cell lines, or different experiment, or *any combinations* of those three. For example, user can color the data by cell line_treatment to compare two (or more) cell lines on different treatments. The possibility is combinatorial now.  

Additionally, it can
- plot image-level boxplots on a selected fitted parameters.

## SDT toolbox
### Single Cell Phasor Plotting 
Copy and paste the path to the folder that contains all your sdt files and masks and one IRF file. This only works in the *offline* mode, as the online app does have access to your local file system. The folder can and is recommended to have sub-folders, each of which contains files for a different treatment. 

### ROI Summing 
Input: a path to a folder that contains sdt files and the associating masks. Output: a folder named `summed_sdt` inside the folder users provided that contains the ROI summed sdt file. 

### SDT fitting 
To be developed. 

## Classification
User uploads a CSV file that has column named `treatment`, and the tool will extract the NADH, FAD, and morphology features that user can choose as features to classify between the treatments. If there is more than two unique treatments, user can choose among all the possible ways to classify. User can also choose how to split the training and test size. Additionally, the tool provides three classification methods: random forest, support vector machine, logistic regression. In all of them, the tool reports the accuracy and plots the confusion matrix and the ROC curve. The tool also plots the random forest's feature importance graph.

## Visualizations
### Feature comparison 
Many times we want to compare a feature (e.g. NADH a1) between different treatments or cell types, and perform statistical tests on that feature. 
The feature comparison module supports comparison of a feature selected by the user and performs pair-wise statistical tests between pairs of treatments/cell types, with the capability of user chosing which pair to conduct the test on. 

## AI toolbox
To be developed. 

# Deployment 
It is deployed on using streamlit's cummunity server at: [https://flim-playground.streamlit.app/](https://flim-playground.streamlit.app/). It is free and will automatically fetch the new commits and update the app. The first time visit will have longer delay because Streamlit is managing in the background to set up machines and a python environment to run this app. 

# Progress
- 12/6/24: bootstrapping the app with very minimal elements. Finished the deployment. 
- 12/9/24: working on designing the modules and the layout of the app. 
- 12/11 - 12/13/24: working on the clustering & outlier finder module. User can interactively select and remove outliers from the visualization and download the outliers-removed dataset. 
- 12/14/24: working on adding filtering mechanism for cluster& outlier finder module to support exploring big datasets. 
- 12/16/24: code refactorization, add the option to color by more than one column. If more than one column is selected to produce colors, different colors are made for each unique combination of unique values of the selected columns (e.g. If your dataset has 3 treatments and 4 cell lines, and you want to colored by treatments and cell_lines, it will create 12 = 3 x 4 colors for each unique combination)
- 12/17/24: verified setting up the required conda environment on mac and windows. Thanks Kayvan for testing in out!
- 12/19/24: add the functionality for removing single cell outliers; add the filter & color_by day option (per Alexa's requests)
- 12/20 - 12/21/24: working on the pipeline of roi summing and then dimension reduction.
- 12/22/24: finished phasor calculation and plotting.
- 12/23/24: added filters for phasor and the dimension reduction on roi sum data method; finished the feature comparison plot with statistical tests. 
- 12/27/24: finished ROI summing, removed the SDT conversion module. 
- 1/3/25: finished the classification module that contains SVM, Logistic Regression, and Random Forest. Did classfication and plotted the confusion matrix, ROC curve and feature importance graph (for random forest only). 
- 3/25/25: started working on FLIM Playground 2.0 
- 3/26/25: refactoring the visulization playground
- 3/27/25: finished classification; started on feature comparison 

# TODO
- add detailed classification reports
- polpulation hetergenity 

# build

streamlit-desktop-app build main.py --name Flim-Playground  --icon logo.png --pyinstaller-options --add-data src:src --add-data pages:pages --add-data config.toml:. --add-data logo.png:. --hidden-import pages.classification --hidden-import pages.visualization --hidden-import pages.data_extraction --collect-all streamlit_plotly_events --onefile --noconfirm --clean

## 
Tested on Mac OS 15, Windows 11