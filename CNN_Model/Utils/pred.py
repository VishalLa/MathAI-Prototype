import cv2 
import torch 
import numpy as np 

from ..Covonutional_neural_network.CNNnetwork import CNN 
from ..Utils.contour import detect_contours, boundaryes

from required_variables import index_to_label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(network, x):
    """
    Predicts the class of a single input using the trained network.

    Args:
        network (nn.Module): The trained model.
        x (torch.Tensor): Input tensor [1,1,64,64].

    Returns:
        str: Predicted character.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    network = network.to(device)

    with torch.no_grad():
        output = network(x)

    print(output)
    predicted_index = torch.argmax(output, dim=1).item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label



def predict_characters(model, canvas_array):

    """
    Detect and predict characters from an input image using a trained model.

    This function:
        1. Converts the input image to grayscale if needed.
        2. Detects contours and extracts character bounding boxes.
        3. Preprocesses each character image and converts it to a tensor.
        4. Runs the model to predict the class for each character.
        5. Returns a list of predicted class indices for all detected characters.

    Args:
        model: Trained PyTorch model for character recognition.
        canvas_array (np.ndarray): Input image as a 2D (grayscale) or 3D (RGB) NumPy array.

    Returns:
        list: List of predicted class indices for each detected character. Returns an empty list if no valid characters are detected.
    """

    model = model.to(device)
    model.eval()

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
