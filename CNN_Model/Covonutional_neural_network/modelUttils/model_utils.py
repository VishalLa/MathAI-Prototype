import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from CNN_Model.Covonutional_neural_network.required_variables import  classes, iterations


def save_model(network, filename='C:\\Users\\visha\\OneDrive\\Desktop\\MathAI\\CNN_Model\\model_parameters_for_CNN.pth'):
    '''
    Saves the model parameters (weights and biases) to a file.

    Parameters:
        network (nn.Module): The PyTorch model to save.
        filename (str): Name of the file to save the parameters.
    '''
    torch.save(network.state_dict(), filename)
    print(f'Model parameters saved to {filename}')



def load_model(network, filename='C:\\Users\\visha\\OneDrive\\Desktop\\MathAI\\CNN_Model\\model_parameters_for_CNN.pth'):
    '''
    Loads the model parameters (weights and biases) from a file.

    Parameters:
        network (nn.Module): The PyTorch model to load the parameters into.
        filename (str): Name of the file to load the parameters from.
    '''
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        state_dict = torch.load(filename, map_location=device)
        
        if 'pos_embed' in state_dict:
            pos_embed = state_dict['pos_embed']
            if pos_embed.shape[1] != network.pos_embed.shape[1]:
                print("Adjusting positional embeddings size...")
                
        network.load_state_dict(state_dict)
        network.eval()
        print(f'Model parameters loaded from {filename}')
        return network
    except Exception as e:
        print(f'Error loading model: {e}')
        raise



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
