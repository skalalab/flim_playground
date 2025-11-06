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
    
    # Vectorized centroid calculation
    centroid_y = np.mean(y_coords)
    centroid_x = np.mean(x_coords)
    
    # Optimize: only calculate distances for masked pixels (avoid full image calculation)
    # This reduces computation significantly for sparse masks
    y_diff = y_coords - centroid_y
    x_diff = x_coords - centroid_x
    mask_distances = np.sqrt(y_diff**2 + x_diff**2)
    max_distance = np.max(mask_distances)
    
    # Handle edge case: if all pixels are at the same location (max_distance == 0)
    # In this case, all intensity is in the innermost ring by definition
    INNERMOST_RING = 1
    if max_distance == 0:
        return 1.0 if ring_number == INNERMOST_RING else 0.0
    
    # Divide into 4 equal rings based on distance
    ring_thickness = max_distance / 4.0
    
    # Define ring boundaries
    ring_min = (ring_number - 1) * ring_thickness
    ring_max = ring_number * ring_thickness
    
    # Create ring mask for pixels only (not full image)
    if ring_number == 4:
        # For the outermost ring, include the maximum distance
        ring_pixel_mask = (mask_distances >= ring_min) & (mask_distances <= ring_max)
    else:
        ring_pixel_mask = (mask_distances >= ring_min) & (mask_distances < ring_max)
    
    # Calculate intensity in this ring (using masked pixels only)
    ring_intensity = np.sum(cell_image[y_coords[ring_pixel_mask], x_coords[ring_pixel_mask]])
    
    # Calculate total intensity in the entire cell
    total_intensity = np.sum(cell_image[mask])
    
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
    
    if len(y_coords) == 0:
        return 0.0
    
    centroid_y = np.mean(y_coords)
    centroid_x = np.mean(x_coords)
    
    # step2: get the intensity weighted centroid of the cell
    # Optimize: extract intensities once
    cell_intensities = cell_image[y_coords, x_coords]
    total_intensity = np.sum(cell_intensities)
    
    if total_intensity == 0:
        return 0.0
    
    # Vectorized weighted centroid calculation
    intensity_weighted_centroid_y = np.sum(y_coords * cell_intensities) / total_intensity
    intensity_weighted_centroid_x = np.sum(x_coords * cell_intensities) / total_intensity
    
    # step3: calculate the geometric displacement between the centroid and the intensity weighted centroid
    # Use np.hypot for more numerically stable distance calculation
    geometric_displacement = np.hypot(centroid_y - intensity_weighted_centroid_y, 
                                     centroid_x - intensity_weighted_centroid_x)
    return geometric_displacement
