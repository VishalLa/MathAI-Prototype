import os
import cv2 
import torch 
import numpy as np
from PIL import Image
from .augment_dataset import add_label_noise


label_to_index = {
    '0': 0,
    '1': 1, 
    '2': 2, 
    '3': 3, 
    '4': 4, 
    '5': 5, 
    '6': 6, 
    '7': 7, 
    '8': 8, 
    '9': 9, 
    'add': 10,
    'sub': 11, 
    'mul': 12, 
    'div': 13, 
    'dec': 14,
    'eq': 15,
    'x': 16, 
    'y': 17, 
    'z': 18,  
    '(': 19, 
    ')': 20,
}

# Reverse mapping for predictions
index_to_label = {v: k for k, v in label_to_index.items()}


def add_padding(image, stride=28):
    """
    Add padding to the image to make it a square with dimensions that are multiples of stride.
    """
    h, w = image.shape[:2]
    max_side = max(h, w)
    new_size = ((max_side + stride - 1) // stride) * stride
    padded_image = np.ones((new_size, new_size), dtype=np.uint8) * 255
    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2
    padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
    return padded_image


def load_mnist_dataset(path):
    data = np.load(path, allow_pickle=True)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    image = np.concatenate((x_train, x_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)

    return image, label



def load_dataset_for_images(folder_path, target_size=(28, 28)):

    custom_images = []
    custom_labels = []
    
    if not os.path.exists(folder_path):
        print(f'Folder {folder_path} dose not exist.')
        return np.array(custom_images), np.array(custom_labels)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)

            try:
                img = Image.open(file_path).convert('L')
                img_array = np.array(img)

                if img_array.dtype != 'uint8':
                    img_array = cv2.convertScaleAbs(img_array)

                _, binary_image =  cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                lower_thresh = max(0, 0.5 * np.median(img_array))
                upper_thresh = min(255, 1.5 * np.median(img_array))
                edges = cv2.Canny(img_array, lower_thresh, upper_thresh)

                # Check if edges are detected
                if np.sum(edges) == 0:
                    print(f"Skipping image {file_path}: No edges detected.")
                    continue

                x, y, z, h = cv2.boundingRect(edges)

                aspect_ratio = z / h
                if z > 5 and h > 5 and 0.2 < aspect_ratio < 1.5:
                    char_image = edges[y:y+h, x:x+z]
                    char_image_resized = cv2.resize(
                        char_image, target_size, interpolation=cv2.INTER_AREA
                    )
                    char_image_resized = add_padding(char_image_resized, stride=4)
                    char_image_resized = cv2.bitwise_not(char_image_resized)
                    char_image = char_image_resized.astype(np.float32) / 255.0
                    
                if '-' in filename:
                    label_str = filename.split('-')[0]
                    if label_str not in label_to_index:
                        print(f'Skipping image {file_path}: Label {label_str} not in label_to_index')
                        continue


                    numeric_label = label_to_index[label_str]
                    binary_image = add_label_noise(char_image, numeric_label)
                else:
                    print(f'Skipping image {file_path}: Unable to extract label')
                    continue
            
                if label_str not in label_to_index:
                    print(f"Skipping image {file_path}: Label '{label_str}' not in label_to_index")
                    continue
                
                custom_images.append(binary_image)
                custom_labels.append(label_to_index[label_str])


            except Exception as e:
                print(f'Error loading image {file_path}: {type(e).__name__} - {e}')
                continue
    return custom_images, custom_labels



def load_dataset(folder_path: list[str])->tuple[np.ndarray, np.ndarray]:

    print('Loading Dataset .............')
    path_for_mnist = 'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\mnist.npz'
    mnist_images, mnist_labels = load_mnist_dataset(path_for_mnist)

    # Normalize MNIST images to [0, 1]
    mnist_images = mnist_images / 255.0

    # Add a channel dimension to MNIST images (for grayscale images)
    mnist_images = mnist_images[:, np.newaxis, :, :]

    images = []
    labels = []

    images.append(mnist_images)
    labels.append(mnist_labels)

    for path in folder_path:
        if not os.path.exists(path):
            print(f"Skipping paths: {path} (Path does not exist)")
            continue
        
        custom_images, custom_labels = load_dataset_for_images(path)

        # Skip if no valid images were found
        if len(custom_images) == 0:
            print(f"No valid images found in folder: {path}")
            continue

        # Add a channel dimension to custom images (for grayscale images)
        custom_images = np.expand_dims(np.array(custom_images), axis=1)


        # Append the images and labels
        images.append(custom_images)
        labels.append(custom_labels)
    
    # Combine all images and labels
    combined_images = np.concatenate(images, axis=0)
    combined_labels = np.concatenate(labels, axis=0)

    # Convert from NumPy to Tensor
    combined_images = torch.tensor(combined_images, dtype=torch.float32)
    combined_labels = torch.tensor(combined_labels, dtype=torch.long)   

    print(f'Shape of imags are: {combined_images.shape}') 

    print(f'Total images lodded: {combined_images.shape[0]}')

    return combined_images, combined_labels
