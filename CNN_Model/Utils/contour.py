import cv2
import torch
import numpy as np


def preprocess_image(canvas_array: np.ndarray):
    if not isinstance(canvas_array, np.ndarray):
        canvas_array = np.array(canvas_array)
    if canvas_array.size == 0:
        raise ValueError("Input array is empty")
    if canvas_array.ndim == 3 and canvas_array.shape[-1] == 3:
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_BGR2GRAY)
    elif canvas_array.ndim == 3 and canvas_array.shape[-1] == 4: 
        canvas_array = cv2.cvtColor(canvas_array, cv2.COLOR_RGBA2GRAY)
    elif canvas_array.ndim != 2:
        raise ValueError("Input array must be 2D (grayscale) or 3D (color image)")

    canvas_array = canvas_array.astype(np.uint8)

    blurred_image = cv2.GaussianBlur(canvas_array, (5, 5), 0)
    cv2.imwrite('blurred_image.png', blurred_image) 

    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite('binary_image.png', binary_image)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    cv2.imwrite('eroded_image.png', eroded_image)

    edges = cv2.Canny(eroded_image, 50, 100)
    cv2.imwrite('edge.png', edges)

    dilated_edges = cv2.dilate(edges, kernel,iterations=2)
    cv2.imwrite('dilated_edges.png', dilated_edges)

    return dilated_edges


def detect_contours(canvas_array):
    edges = preprocess_image(canvas_array)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def add_padding(image, stride=4):
    h, w = image.shape[:2]
    max_side = max(h, w)
    new_size = ((max_side + stride - 1) // stride) * stride
    padded = np.ones((new_size, new_size), dtype=np.uint8) * 255
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return padded


def boundaryes(contours, gray):
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h

        if area < 30 or w < 5 or h < 5 or not (0.1 < aspect_ratio < 10):
            continue

        char_crop = gray[y:y+h, x:x+w]
        coords = cv2.findNonZero(char_crop)
        if coords is None:
            continue

        x_, y_, w_, h_ = cv2.boundingRect(coords)
        char_crop = char_crop[y_:y_+h_, x_:x_+w_]

        char_crop = add_padding(char_crop, stride=4)
        char_crop = cv2.resize(char_crop, (64, 64), interpolation=cv2.INTER_AREA)

        _, char_bin = cv2.threshold(char_crop, 127, 255, cv2.THRESH_BINARY)
        char_bin = char_bin.astype(np.float32) / 255.0

        bounding_boxes.append((x, char_bin))

    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])  # left to right
    return bounding_boxes
