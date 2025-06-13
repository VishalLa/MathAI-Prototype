import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import copy
from joblib import Parallel, delayed

from Covonutional_neural_network.modelUttils.loaddataset import load_dataset
from Covonutional_neural_network.modelUttils.model_utils import save_model, test, train, split_dataset

from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from Covonutional_neural_network.CNNnetwork import CNN, FocalLoss
from Covonutional_neural_network.ViTnetwork import ViT

from required_variables import learning_rate, classes

# Add Covonutional_neural_network path to model 

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
sys.path.append(current_directory)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the model architecture
network = CNN().to(device)

# network = ViT(
#     img_size=64,
#     patch_size=8,
#     in_channels=1,
#     num_classes=classes,
#     emb_size=384,
#     depth=8,
#     n_heads=12,
#     mlp_dim=764
# ).to(device)


class ModelPipeline:
    def __init__(self, data_folder: list[str]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = network

        self.loss_fn = nn.CrossEntropyLoss()
        self.data_folder = data_folder

        # Load and prepare dataset
        self.X, self.Y = load_dataset(
            folder_path = self.data_folder
        )

        print(f"Unique labels in dataset: {torch.unique(self.Y)}")
        self.X_train, self.X_test, self.Y_train, self.Y_test = split_dataset(self.X, self.Y)

        self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train, random_state=101)

        self.X_train = self.X_train.clone().detach().to(torch.float32).view(-1, 1, 64, 64).to(self.device)
        self.X_test = self.X_test.clone().detach().to(torch.float32).view(-1, 1, 64, 64).to(self.device)
        self.Y_train = self.Y_train.clone().detach().to(torch.long).to(self.device)
        self.Y_test = self.Y_test.clone().detach().to(torch.long).to(self.device)


    def train_model(self, data_loader, loss_function, optimizer):
        train(
            network=self.model,
            data_loader=data_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=self.device
        )


    def _train_and_validate_fold(self, train_idx, val_idx, batch_size, learning_rate, fold, k):
        # Each fold gets its own model instance
        
        model = copy.deepcopy(self.model).to(self.device)

        criterion = FocalLoss(gamma=2.0, alpha=1.0).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        train_images, val_images = self.X_train[train_idx], self.X_train[val_idx]
        train_labels, val_labels = self.Y_train[train_idx], self.Y_train[val_idx]

        train_dataset = TensorDataset(train_images, train_labels)
        val_dataset = TensorDataset(val_images, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train
        train(
            network=model,
            data_loader=train_loader,
            loss_function=criterion,
            optimizer=optimizer,
            device=self.device
        )

        # Validate
        model.eval()
        val_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if torch.max(targets) >= classes or torch.min(targets) < 0:
                    print(f"Invalid target label found in validation: {torch.max(targets)}, shape: {targets.shape}")
                    continue
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                num_batches += 1

        accuracy = 100 * correct / total if total > 0 else 0
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('nan')
        print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")
        return avg_val_loss, accuracy, model if fold == k - 1 else None


    def cross_validate_model(self, k=5, batch_size=64, learning_rate=learning_rate, n_jobs=5):
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(kfold.split(self.X_train))

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._train_and_validate_fold)(train_idx, val_idx, batch_size, learning_rate, fold, k)
            for fold, (train_idx, val_idx) in enumerate(splits)
        )

        fold_losses = [r[0] for r in results]
        fold_accuracies = [r[1] for r in results]
        final_model = [r[2] for r in results if r[2] is not None][0]
        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        avg_error = 1 - (avg_accuracy / 100)
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print(f"Average Error: {avg_error:.4f}")

        self.model = final_model
        return fold_losses, fold_accuracies, avg_accuracy, avg_error

    def test_model(self):
        test(self.model, self.X_test, self.Y_test)

    def save_model(self):
        save_model(self.model)
