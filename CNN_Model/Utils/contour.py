import cv2 
import torch
import numpy as np 


def add_padding(image: np.ndarray, stride: int=4) -> np.ndarray:
    """
    Add white padding to make the input image square and its size a multiple of the given stride.

    Args:
        image (np.ndarray): Input grayscale image as a 2D NumPy array.
        stride (int, optional): The stride to which the output size should be aligned. Default is 4.

    Returns:
        np.ndarray: Padded square image with side length as a multiple of stride.
    """
    h, w = image.shape[:2]
    max_side = max(h, w)
    new_size = ((max_side + stride - 1) // stride) * stride
    padded = np.zeros((new_size, new_size), dtype=np.uint8) * 255
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return padded


def preprocess_image(canvas_array: np.ndarray) -> np.ndarray:

    """
    Preprocess an input image array for contour detection.

    This function takes a grayscale or color image as a NumPy array, converts it to grayscale if needed,
    applies Gaussian blur, adaptive thresholding, erosion, Canny edge detection, and dilation to produce
    a binary edge map suitable for contour extraction.

    Steps performed:
        1. Converts color images (BGR or RGBA) to grayscale.
        2. Ensures the image is 8-bit unsigned integer type.
        3. Applies Gaussian blur to reduce noise.
        4. Applies adaptive thresholding to binarize the image.
        5. Erodes the binary image to remove small noise.
        6. Detects edges using the Canny algorithm.
        7. Dilates the edges to strengthen them.

    Args:
        canvas_array (np.ndarray): Input image as a 2D (grayscale) or 3D (color) NumPy array.

    Returns:
        np.ndarray: Processed binary image with dilated edges.

    Raises:
        ValueError: If the input array is empty or not a valid image format.
    """

    if canvas_array.size == 0:
        raise ValueError("Input array is empty !")
    
    if canvas_array.ndim == 3 and canvas_array.shape[-1] == 3:
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_BGR2GRAY)

    elif canvas_array.ndim == 3 and canvas_array.shape[-1] == 4:
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_RGBA2GRAY)

    elif canvas_array.ndim != 2:
        raise ValueError("Input array must be 2D (grayscale) or 3D (color image)")
    
    # Convert canvas_array in 8 bit channel 
    canvas_array = canvas_array.astype(np.uint8)

    blurred_image = cv2.GaussianBlur(canvas_array, (5, 5), 0)
    
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2 
    )
    cv2.imwrite('binary_image.png', binary_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    cv2.imwrite('eorded_image.png', eroded_image)
    
    edges = cv2.Canny(eroded_image, 50, 100)
    cv2.imwrite('edge.png', edges)

    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    cv2.imwrite('dilated_edges.png', dilated_edges)

    return dilated_edges



def detect_contours(canvas_array: np.ndarray) -> list[np.ndarray]:
    """
    Detect contours in a preprocessed image.

    This function preprocesses the input image and then finds external contours.

    Args:
        canvas_array (np.ndarray): Input image as a 2D or 3D NumPy array.

    Returns:
        list[np.ndarray]: List of contours, each as a NumPy array of points.
    """

    edges = preprocess_image(canvas_array)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def boundaryes(contours: list[np.ndarray], gray: np.ndarray) -> list[np.ndarray]:

    """
    Extract and process bounding boxes for valid contours from a grayscale image.

    For each contour, this function computes the bounding rectangle, filters out small or invalid regions,
    crops and pads the character region, resizes it to 64x64, binarizes it, and collects the result.

    Args:
        contours (list[np.ndarray]): List of contours (as returned by detect_contours).
        gray (np.ndarray): Grayscale image from which to extract character crops.

    Returns:
        list[np.ndarray]: List of tuples (x, char_bin), where x is the x-coordinate of the bounding box
                          and char_bin is the processed character image (float32, shape 64x64).
    """
    
    bounding_boxes = []

    height, width = gray.shape

    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, contours, -1, (0,255,0), 2)
    cv2.imwrite('contours_debug.png', debug_img)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_area = max(areas)

    for contour, area in zip(contours, areas):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w/h 

        # Skip larges are or contour touching the edge
        if area == max_area or x <= 1 or y <= 1 or x+w >= width - 1 or y+h >= height-1:
            continue

        # Skip border contour (if it's almost as big as the image)
        if w > 0.9 * width and h > 0.9 * height:
            continue

        # Filter out small or unlikely regions
        if area < 9 or w < 4 or h < 4 or not (0.15 < aspect_ratio < 10):
            continue

        char_crop_1 = gray[y:y+h, x:x+w]
        coords = cv2.findNonZero(char_crop_1)
        if coords is None:
            continue

        x_, y_, w_, h_ = cv2.boundingRect(coords)

        char_crop_2 = char_crop_1[y_:y_+h_, x_:x_+w_]

        char = add_padding(char_crop_2)
        char = cv2.resize(char, (64, 64), interpolation=cv2.INTER_AREA)

        _, char_bin = cv2.threshold(char, 127, 255, cv2.THRESH_BINARY)
        char_bin = char_bin.astype(np.float32) / 255.0

        bounding_boxes.append((x, char_bin))

    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    # print(bounding_boxes)
    
    for j, (_, i) in enumerate(bounding_boxes):
        if i.dtype != np.uint8:
            img_to_save = (i * 255).astype(np.uint8)
        else:
            img_to_save = i
        # print(f"Saving char{j+1}.png: dtype={img_to_save.dtype}, min={img_to_save.min()}, max={img_to_save.max()}")
        # cv2.imwrite(f'char{j+1}.png', img_to_save)

    return bounding_boxes

