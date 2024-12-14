

outlierFinder = """
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
    - Some outliers are not caused by errors using SPCImage. Instead, it may be due to inconsistencies occurred during image/data acquisition or masking. Therefore, PCA can be performed on input data (SDTs and Masks) directly: each cell occupies a set of spatial pixels, under each of which is 256 time bins. We can sum the times spatially to get the cell-level 256 time bins and perform PCA on them. 

### Design 
You can identify the outliers by hovering over the points that show the `base_name` (we assume you dataset has a column called `base_name`, which is of this format: `image_name` + `_` + `cell_number`). You can select the outlier by clicking on it, then you will have the option to remove it and cells that belongs to the same image. Then the algorithm and visualization will be rerun on the new data without the outlier image. The rationale is that if all the outlier points belong to the same image, them that image should probably be reanalyzed. 

For big datasets such as the redox dataset, which has multiple experiments, cell lines, treatments, and media, it offers filtering based on those columns. 

Additionally, it can
- plot image-level boxplots on a selected fitted parameters.
"""

sdtSuite = """
## Overview 
It features a list tools that takes SDT files as inputs and perform operations on them. 

## Tools
### ROI Summing 

### SDT fitting 
### Phasor Analysis 
The phasor approach is an alternative, quick, and fit-free method for analyzing FLIM data. It is based on the Fourier transformation of the fluorescence decay curve. 
The phasor plot is a scatter plot of the real and imaginary components of the Fourier transformation of the fluorescence decay curve. It can be used to identify different fluorophores in a sample, to distinguish between free and bound states of a fluorophore, and to detect changes in the microenvironment of a fluorophore.

#### Workflow
##### Inputs 
SDT files and the corresponding masks, and IRF. 

##### Steps
1. 

##### Outputs
Single cell phasor coordinates; singke cell phasor plots. 
"""


classification = """to be developed."""
regionProps = """to be developed. """
plotting = """to be developed. """