import codecs
import numpy as np
import pathlib
from pathlib import Path
import tifffile
from typing import Union

def load_image(path: Union[str, pathlib.PurePath]) -> np.ndarray:
    """
    Detects the extension and loads image into a numpy array 
    if it's a tif/tiff or an asc file.

    Parameters
    ----------
    path : pathlib path or str
        path to the image.

    Returns
    -------
    np.ndarray
        ndarray with the image data.

    """
    if not isinstance(path, pathlib.PurePath):
        path = Path(path)
    pass
    if path.suffix == ".asc":
        return read_asc(path)
    if path.suffix in [".tiff", ".tif"]:
        return tifffile.imread(path)
    
def read_asc(path):
    """
    Reads in an asc file into a numpy ndarray

    Parameters
    ----------
    path : pathlib.Path
        path to the file.

    Returns
    -------
    array : np.ndarray
        Numpy array holding the image data.

    """
    with codecs.open(path, encoding="utf-8-sig") as file:
        # for each line for each value, convert to float
        array = np.array([[float(x) for x in line.split()] for line in file])

    return array