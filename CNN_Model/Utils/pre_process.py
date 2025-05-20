import cv2
import torch
import numpy as np
from ..Covonutional_neural_network.CNNnetwork import CNN
from ..Covonutional_neural_network.ViTnetwork import ViT

from ..Covonutional_neural_network.modelUttils.model_utils import predict
from ..Utils.contour import detect_contours, boundaryes


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

    resized_img = cv2.resize(padded_image, (64, 64), interpolation=cv2.INTER_AREA)

    return resized_img

label_to_index = {
    '0': 0,
    '1': 1, 
    '2': 2, 
    '3': 3, 
    '4': 4, 
    '5': 5, 
    '6': 6, 
    # '7': 7, 
    # '8': 8, 
    # '9': 9, 
    # 'add': 10,
    # 'dec': 11, 
    # 'div': 12, 
    # 'eq': 13, 
    # 'mul': 14,
    # 'sub': 15,
    # # '(': 16, 
    # # ')': 17, 
    # 'x': 16,  
    # 'y': 17, 
    # # 'z': 20,
}

# Reverse mapping for predictions 

index_to_label = {v: k for k, v in label_to_index.items()}

def predict_charheacters(model, canvas_array):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    # Convert to grayscale if not already
    if len(canvas_array) == 3:
        gray = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = canvas_array

    contours = detect_contours(gray)
    bounding_boxes = boundaryes(contours, gray)

    predictions = []

    if not bounding_boxes:
        print("No valid characters detected.")
        return []

    for _, img in bounding_boxes:
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        pred = predict(network=model, x=img_tensor)
        predictions.append(pred)

    # chars = [index_to_label[i] for i in predictions]

    return predictions
