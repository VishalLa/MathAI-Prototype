import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

classes = 7
iterations = 15

def save_model(network, filename='C:\\Users\\visha\\OneDrive\\Desktop\\MathAI\\CNN_Model\\model_parameters.pth'):
    '''
    Saves the model parameters (weights and biases) to a file.

    Parameters:
        network (nn.Module): The PyTorch model to save.
        filename (str): Name of the file to save the parameters.
    '''
    torch.save(network.state_dict(), filename)
    print(f'Model parameters saved to {filename}')



def load_model(network, filename='C:\\Users\\visha\\OneDrive\\Desktop\\MathAI\\CNN_Model\\model_parameters.pth'):
    '''
    Loads the model parameters (weights and biases) from a file.

    Parameters:
        network (nn.Module): The PyTorch model to load the parameters into.
        filename (str): Name of the file to load the parameters from.
    '''
    state_dict = torch.load(filename)
    network.load_state_dict(state_dict)
    network.eval()  # Set the model to evaluation mode
    print(f'Model parameters loaded from {filename}')
    return network



def split_dataset(X, Y, size=0.9):
    train_size = int(size*X.shape[0])
    test_size = X.shape[0] - train_size
    print(f'train size: {train_size}')
    print(f'test size: {test_size}')

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X = X[indices]
    Y = Y[indices]

    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    return X_train, X_test, Y_train, Y_test



def train(network, data_loader, loss_function, optimizer, device, epochs=iterations, batch_size=64):
    
    network.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for x_batch, labels_batch in data_loader:
            # Ensure the data is on the correct device (GPU/CPU)
            x_batch, labels_batch = x_batch.to(device), labels_batch.to(device)

            if torch.max(labels_batch) >= classes:  
                print(f"Invalid target label found: {torch.max(labels_batch)}, shape: {labels_batch.shape}")
                continue

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = network(x_batch)

            # Calculate loss
            loss = loss_function(outputs, labels_batch)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Print loss after each epoch
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}')



def test(network, X_test, Y_test, batch_size=64):
    '''
    Tests the network on the test dataset.

    Parameters:
        network (nn.Module): The trained network.
        X_test (torch.Tensor): Test inputs.
        Y_test (torch.Tensor): Test labels.
        batch_size (int): Batch size for testing.

    Returns:
        None
    '''
    # Create a DataLoader for the test dataset
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.eval()  # Set the network to evaluation mode

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, labels_batch in test_loader:
            # Ensure the data is on the correct device (GPU/CPU)
            x_batch, labels_batch = x_batch.to(device), labels_batch.to(device)

            # Forward pass
            outputs = network(x_batch)
            _, predicted = torch.max(outputs.data, 1)

            # Update total and correct counts
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')



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
