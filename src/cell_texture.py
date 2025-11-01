import numpy as np
from skimage import morphology

def granularity(cell_image, n):
    """
    Calculate the granularity of the image using morphological opening
    
    Parameters:
    - cell_image: 2D numpy array (cell ROI intensity image)
    - n: int (radius of disk structuring element)
    
    Returns:
    - granularity value: percentage of intensity removed by opening
    """
    
    # Create circular structuring element of size n
    if n == 1:
        # For size 1, use a simple 3x3 cross
        selem = np.array([[0, 1, 0], 
                         [1, 1, 1], 
                         [0, 1, 0]], dtype=bool)
    else:
        # For larger sizes, use disk structuring element
        selem = morphology.disk(n)
    
    # Apply morphological opening (erosion followed by dilation) to remove bright objects of radius n
    opened_image = morphology.opening(cell_image, selem)
    
    # Calculate intensity difference
    intensity_removed = cell_image - opened_image
    
    # Calculate total intensity in masked region
    total_intensity = np.sum(cell_image)
    
    # Avoid division by zero
    if total_intensity == 0:
        return 0.0
    
    # Calculate percentage of intensity removed
    granularity_value = (np.sum(intensity_removed) / total_intensity) * 100
    
    return granularity_value

def radial_distribution(cell_image, ring_number):
    """
    Calculate the radial distribution mean fraction for a specific ring
    
    Parameters:
    - cell_image: 2D numpy array (intensity image)
    - ring_number: int (1-4, where 1 is innermost, 4 is outermost)
    
    Returns:
    - mean_fraction: fraction of total intensity in the specified ring
    """
    # Find the centroid of the mask
    mask = cell_image > 0
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0:
        return 0.0
    
    centroid_y = np.mean(y_coords)
    centroid_x = np.mean(x_coords)
    
    # Create coordinate arrays
    y_indices, x_indices = np.indices(cell_image.shape)
    
    # Calculate distance from centroid to each pixel
    distances = np.sqrt((y_indices - centroid_y)**2 + (x_indices - centroid_x)**2)
    
    # Only consider distances within the mask
    mask_distances = distances[mask]
    max_distance = np.max(mask_distances)
    
    # Divide into 4 equal rings based on distance
    ring_thickness = max_distance / 4.0
    
    # Define ring boundaries
    ring_min = (ring_number - 1) * ring_thickness
    ring_max = ring_number * ring_thickness
    
    # Create ring mask
    ring_mask = (distances >= ring_min) & (distances < ring_max) & mask
    
    # For the outermost ring (ring 4), include the maximum distance
    if ring_number == 4:
        ring_mask = (distances >= ring_min) & (distances <= ring_max) & mask
    
    # Calculate intensity in this ring
    ring_intensity = np.sum(cell_image[ring_mask])
    
    # Calculate total intensity in the entire cell
    total_intensity = np.sum(cell_image)
    
    # Avoid division by zero
    if total_intensity == 0:
        return 0.0
    
    # Calculate fraction of total intensity in this ring
    mean_fraction = ring_intensity / total_intensity
    
    return mean_fraction

def mass_displacement(cell_image):
    # geometric displacement between the centroid and the intensity weighted centroid of the cell 
    # step1: get the centroid of the cell
    cell_mask = cell_image > 0
    y_coords, x_coords = np.where(cell_mask)
    centroid_y = np.mean(y_coords)
    centroid_x = np.mean(x_coords)
    # step2: get the intensity weighted centroid of the cell
    intensity_weighted_centroid_y = np.sum(y_coords * cell_image[y_coords, x_coords]) / np.sum(cell_image[y_coords, x_coords])
    intensity_weighted_centroid_x = np.sum(x_coords * cell_image[y_coords, x_coords]) / np.sum(cell_image[y_coords, x_coords])
    # step3: calculate the geometric displacement between the centroid and the intensity weighted centroid
    geometric_displacement = np.sqrt((centroid_y - intensity_weighted_centroid_y)**2 + (centroid_x - intensity_weighted_centroid_x)**2)
    return geometric_displacement
