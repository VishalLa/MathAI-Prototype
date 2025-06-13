import cv2
import torch
import numpy as np

from ..Utils.contour import add_padding

def prepare_canvas(canvas_array: np.ndarray)->np.ndarray:

    """
    Prepare a canvas image for further processing or model input.

    This function:
        1. Checks if the input array is valid and non-empty.
        2. Converts RGBA or RGB images to grayscale.
        3. Applies binary inverse thresholding to create a binary image.
        4. Pads the binary image to make it square and its size a multiple of the given stride.

    Args:
        canvas_array (np.ndarray): Input image as a 2D (grayscale) or 3D (RGB/RGBA) NumPy array.

    Returns:
        np.ndarray: Padded binary image suitable for further processing.

    Raises:
        ValueError: If the input array is empty or None.
    """

    if canvas_array is None or canvas_array.size == 0:
        raise ValueError('Canvas array is empty or None !')
    
    if canvas_array.shape[-1] == 4:
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_RGBA2GRAY)
    elif canvas_array.shape[-1] == 3:
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)

    _, binary_image = cv2.threshold(canvas_array, 127, 255, cv2.THRESH_BINARY_INV)

    padded_image = add_padding(binary_image, stride=20)

    return padded_image