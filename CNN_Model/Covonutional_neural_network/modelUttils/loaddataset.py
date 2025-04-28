import os
import cv2 
import torch 
import numpy as np
from PIL import Image


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
    'plus': 10,
    'minus': 11, 
    'slash': 12, 
    'dot': 13, 
    'w': 14, 
    'x': 15, 
    'y': 16, 
    'z': 17,  
    '(': 18, 
    ')': 19,
}

# Reverse mapping for predictions
index_to_label = {v: k for k, v in label_to_index.items()}


def load_mnist_dataset(path):
    data = np.load(path, allow_pickle=True)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    image = np.concatenate((x_train, x_test), axis=0)
    label = np.concatenate((y_train, y_test), axis=0)

    return image, label

def load_dataset_from_folder(folder_path, target_size=(28, 28)):

    custom_images = []
    custom_labels = []
    
    if not os.path.exists(folder_path):
        print(f'Folder {folder_path} dose not exist.')
        return custom_images, custom_labels
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)

            try:
                img = Image.open(file_path).convert('L')
                img = img.resize(target_size)
                img_array = np.array(img)
                _, binary_image =  cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                if '-' in filename:
                    label_str = filename.split('-')[0]
                else:
                    print(f'Skipping image {file_path}: Unaple to extract label')
                    continue
            
                if label_str not in label_to_index:
                    print(f"Skipping image {file_path}: Label '{label_str}' not in label_to_index")
                    continue
                
                custom_images.append(binary_image)
                custom_labels.append(label_to_index[label_str])

            except Exception as e:
                print(f'Error loading image {file_path}: {e}')
                continue
    return custom_images, custom_labels


def load_dataset(folder_path: list[str]):
    path_for_mnist = 'C:\\Users\\visha\\OneDrive\\Desktop\\MathAI\\Model\\entiredataset\\mnist.npz'
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
        custom_images, custom_labels = load_dataset_from_folder(path)

        # Skip if no valid images were found
        if len(custom_images) == 0:
            print(f"No valid images found in folder: {path}")
            continue

        # Normalize custom images to [0, 1]
        custom_images = np.array(custom_images) / 255.0

        # Add a channel dimension to custom images (for grayscale images)
        custom_images = custom_images[:, np.newaxis, :, :]

        # Append the images and labels
        images.append(custom_images)
        labels.append(custom_labels)
    
    # Combine all images and labels
    combined_images = np.concatenate(images, axis=0)
    combined_labels = np.concatenate(labels, axis=0)

    # Convert from NumPy to Tensor
    combined_images = torch.tensor(combined_images, dtype=torch.float32)
    combined_labels = torch.tensor(combined_labels, dtype=torch.long)    

    return combined_images, combined_labels
