import cv2
import torch
import numpy as np
from ..Covonutional_neural_network.modelUttils.model_utils import predict

def preprocess_image(canvas_array):
    """
    Preprocess the canvas array to separate connected components using
    erosion, distance transform, and watershed algorithm.

    Parameters:
        canvas_array (np.ndarray): The canvas state as a NumPy array.

    Returns:
        np.ndarray: The edges of the processed image.
    """
    
    # Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(canvas_array, (7, 7), 0)
    cv2.imwrite("blurred_image.png", blurred_image)


    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11 ,2
    )
    cv2.imwrite("binary_image.png", binary_image)


    # Erosion to break apart connected components 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    eroded_image = eroded_image.astype(np.uint8)  # Ensure eroded image is uint8
    cv2.imwrite("eroded_image.png", eroded_image)

    # Distance transform
    dist_transform = cv2.distanceTransform(eroded_image, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("distance_transform.png", dist_transform)

    # Threshold the distance transform to find sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)  # Ensure sure_fg is uint8
    cv2.imwrite("sure_foreground.png", sure_fg)

    # Sure background using dilation
    sure_bg = cv2.dilate(eroded_image, kernel, iterations=2)
    sure_bg = sure_bg.astype(np.uint8)  # Ensure sure_bg is uint8
    cv2.imwrite("sure_background.png", sure_bg)

    # Subtract sure foreground from sure background to get unknown regions
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite("unknown_regions.png", unknown)

    # Marker labelling for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Increment all labels by 1 to ensure background is not 0
    markers[unknown == 255] = 0  # Mark unknown regions as 0

    # Watershed algorithm
    markers = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)
    binary_image[markers == -1] = 0  # Mark boundaries
    cv2.imwrite("watershed_result.png", binary_image)

    # Canny edge detection to get edges
    edges = cv2.Canny(eroded_image, 50, 100)
    cv2.imwrite("edges.png", edges)

    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite('dilated_edges.png', dilated_edges)

    return dilated_edges


def detect_contours(canvas_array):
    """
    Detects contours in the given canvas array.

    Parameters:
        canvas_array (np.ndarray): The canvas state as a NumPy array.

    Returns:
        np.ndarray: The canvas with contours drawn on it.
    """

    edges = preprocess_image(canvas_array)
    
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def boundaryes(contours, gray):
    bounding_boxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h 
        if w > 5 and h > 5 and 0.2 < aspect_ratio < 1.5: # adjust aspect ratio range 
            char_imag = gray[y:y+h, x:x+w]
            char_imag = cv2.resize(char_imag, (28, 28), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite('resized images.png', char_imag)

            # Apply binary threshold to ensure the resized image is binary
            _, char_image = cv2.threshold(char_imag, 127, 255, cv2.THRESH_BINARY)

            char_image = char_image / 255.0
            bounding_boxes.append((x, char_image))

    # sort charachers left to right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    for i, (x, char_image) in enumerate(bounding_boxes):
        cv2.imwrite(f'char_{i}.png', char_image * 255)
        bounding_boxes[i] = (x, char_image)


    return bounding_boxes


def predict_chracters(char_images, model):
    """
    Predict characters from a list of character images.

    Args:
        char_images (list): List of 28x28 np.array (normalized to 0-1).
        model: Loaded CNN model.

    Returns:
        List of predicted labels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    predictions = []

    for _, img in char_images:
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
        img_tensor = img_tensor.to(device)

        pred = predict(model, img_tensor)
        predictions.append(pred)

    return predictions

