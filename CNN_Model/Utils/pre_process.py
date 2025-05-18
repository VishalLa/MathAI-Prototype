import cv2
import numpy as np
from ..Covonutional_neural_network.network import CNN
from ..Utils.contour import detect_contours, boundaryes, predict_chracters


def prepare_canvas(canvas_array):
    if canvas_array is None or canvas_array.size == 0:
        raise ValueError('Canvas array is empty or None.')

    # Step 1: Convert RGBA -> Grayscale
    if canvas_array.shape[-1] == 4:  # If RGBA
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_RGBA2GRAY)
    elif canvas_array.shape[-1] == 3:  # If RGB
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)

    # Step 2: Threshold (binarize) the image
    _, binary_image = cv2.threshold(canvas_array, 127, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Pad the image (make it square)
    h, w = binary_image.shape
    max_side = max(h, w)
    padded_image = np.ones((max_side, max_side), dtype=np.uint8) * 255
    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2
    padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = binary_image

    # Step 4: Resize to 28x28
    resized = cv2.resize(padded_image, (28, 28), interpolation=cv2.INTER_NEAREST)

    return resized



def predict_chars(model, canvas_array):
    """
    Preprocesses the canvas array to prepare it for contour detection.

    Parameters:
        canvas_array (np.ndarray): The canvas state as a NumPy array.

    Returns:
        np.ndarray: The preprocessed canvas.
    """
    # Convert to grayscale if not already
    if len(canvas_array.shape) == 3:
        gray = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = canvas_array

    contours = detect_contours(gray)
    bounding_boxes = boundaryes(contours, gray)
    char_array = predict_chracters(bounding_boxes, model)

    return char_array
