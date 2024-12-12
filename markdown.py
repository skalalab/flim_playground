

outlierFinder = """
## Overview 
This tools performs **clustering** on your data (fitted data or raw data) using dimension reduction algorithms (**PCA** and **UMAP**). 

After clustering it can be used to identify outliers.  
Specifically, the tool provides 3 ways to identify outliers: 
- perform dimension reduction on fitted parameters
- perform dimension reduction on raw inputs (sdt and mask)
- plot image level features 

### 1. Dimension Reduction on Fitted Parameters
#### Motivation 
Fitting Lifetime images using SPCImage can be error-prone, especially when analyzing big datasets on one's own. Therefore, a sanity check method is in need. The tools uses principal component analysis on your uploaded datasets using selected lifetime and morphological variables. In the online setting, PCA is preferred over UMAP because of its speed.

### 2. Dimension Reduction on Raw Inputs
#### Motivation
Some outliers are not caused by errors using SPCImage. Instead, it may be due to inconsistencies occurred during image/data acquisition or masking. Therefore, dimension reduction methods (PCA and UMAP) can be performed on input data (SDTs and Masks) directly: each cell occupies a set of spatial pixels, under each of which is 256 time bins. We can sum the time bins spatially to get the cell-level 256 time bins and perform PCA on them.

#### Design
In the above two ways, user can identify the outliers by hovering over the points that show the `base_name` (we assume you dataset has a column called `base_name` which has the image name and cell number). The rationale is that if all the outlier points belong to the same image, them that image should probably be reanalyzed or inspected or kicked out.

### 3. Image Level Features
#### Motivation
This is another way to find fitting errors. A boxplot is plotted over all images/ROIs of you dataset for the selected feature (lifetimes and intensities and morphologies). 
The `image_name` column is the y-axis, which is inferred from the `base_name` column (it assumes `base_name` = `image_name` + "_cell_number"), and the selected feature is the x-axis. 

"""

sdtSuite = """
## Overview 
It features a list tools that takes SDT files as inputs and perform operations on them. 

## Tools

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


classification = """"""
regionProps = """"""
