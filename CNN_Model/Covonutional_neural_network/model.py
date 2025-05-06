import os
import sys

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modelUttils.loaddataset import load_dataset
from modelUttils.model_utils import save_model, test, train, split_dataset

from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from network import CNN

# Add Covonutional_neural_network path to model 

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
sys.path.append(current_directory)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the model architecture
network = CNN().to(device)


class ModelPipeline:
    def __init__(self, data_folder: list[str]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.data_folder = data_folder

        # Load and prepare dataset
        self.X, self.Y = load_dataset(
            data_foldel = self.data_folder
        )

        print(f"Unique labels in dataset: {torch.unique(self.Y)}")
        self.X_train, self.X_test, self.Y_train, self.Y_test = split_dataset(self.X, self.Y)

        self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train, random_state=42)

        self.X_train = self.X_train.clone().detach().to(torch.float32).view(-1, 1, 28, 28).to(self.device)
        self.X_test = self.X_test.clone().detach().to(torch.float32).view(-1, 1, 28, 28).to(self.device)
        self.Y_train = self.Y_train.clone().detach().to(torch.long).to(self.device)
        self.Y_test = self.Y_test.clone().detach().to(torch.long).to(self.device)

    def train_model(self, data_loder, loss_finction, optimizer):
        train(
            network=self.model,
            data_loder=data_loder,
            loss_function=loss_finction,
            optimizer=optimizer,
            device=self.device
        )

    def cross_validate_model(self, k=5, batch_size=64, learning_rate=0.0001):
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_accuracies = []
        final_model = None

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
            print(f"Fold {fold + 1}/{k}")

            train_images, val_images = self.X_train[train_idx], self.X_train[val_idx]
            train_labels, val_labels = self.Y_train[train_idx], self.Y_train[val_idx]

            train_dataset = TensorDataset(train_images, train_labels)
            val_dataset = TensorDataset(val_images, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.model = CNN().to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            self.train_model(
                data_loder=train_loader,
                loss_finction=criterion,
                optimizer=optimizer
            )

            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    if torch.max(targets) >= 20:
                        print(f"Invalid target label found in validation: {torch.max(targets)}")
                        continue

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            accuracy = 100 * correct / total
            fold_accuracies.append(accuracy)
            print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")

            if fold == k - 1:
                final_model = self.model

        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        avg_error = 1 - (avg_accuracy / 100)
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print(f"Average Error: {avg_error:.4f}")

        self.model = final_model

    def test_model(self):
        test(self.model, self.X_test, self.Y_test)

    def save_model(self):
        save_model(self.model)


data_folder = [
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\0', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\1', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\2', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\3', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\4', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\5', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\6', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\7', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\8', 
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\9',
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\add','C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\symbols\\+','C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\dec','C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\div',
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\eq',    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\mul', 'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\symbols\\x','C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\sub','C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\symbols\\-',
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\x',
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\y',
    'C:\\Users\\visha\\OneDrive\\Desktop\\entiredataset\\dataset\\z'
]


pipeline = ModelPipeline(data_folder=data_folder)

pipeline.cross_validate_model()
pipeline.test_model()

pipeline.save_model()

