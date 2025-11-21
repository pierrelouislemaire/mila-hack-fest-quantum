import matplotlib.pyplot as plt
import numpy as np 
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import *

def set_up_gpus():
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("No GPU available, using CPU.")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42,):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)


def load_timeseries(train_dir, val_dir,):
    print("Loading the input timeseries.")
    imput_train_path = train_dir / "input_series.npy"
    input_val_path = val_dir / "input_series.npy"
    input_train = np.load(imput_train_path).astype(np.float32)
    input_val = np.load(input_val_path).astype(np.float32)
    print(f"The timeseries shape is: {input_train.shape} and validation shape is: {input_val.shape}")

    print("Loading the target timeseries.")
    target_train_path = train_dir / "target_series.npy"
    target_val_path = val_dir / "target_series.npy"
    target_train = np.load(target_train_path).astype(np.float32)
    target_val = np.load(target_val_path).astype(np.float32)
    print(f"The target shape is: {target_train.shape} and validation shape is: {target_val.shape}")

    return (input_train, input_val, target_train, target_val,)

def create_input_target_pairs(input_train, target_train, input_val, target_val,):
    train_dataset = TensorDataset(
        input_train,
        target_train,
    )
    val_dataset = TensorDataset(
        input_val,
        target_val,
    )

    return (train_dataset, val_dataset,)

def data_loader(train_dataset, val_dataset, batch_size,):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return (train_loader, val_loader,)

class dense_model(nn.Module): #TODO
    def __init__(self, input_size=10, hidden_size=50, output_size=1,):
        super(dense_model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    timer = time.time()

    ############################
    #          Set-up          #
    ############################
    # 1. Set GPUs usage.
    device = set_up_gpus()
    # 2. Set random seed for reproducibility.
    set_seed(42)

    ###############################
    #      Paths generation       #
    ###############################
    train_dir = "todo"
    val_dir = "todo"
    test_name = "test1"
    results_dir = f"results/{test_name}"
    # Hyper-parameters
    nn_config = {
        'context_length': 12,
        'batch_size': 64,
        'learning_rate': 1e-03,
        'num_epochs': 100,
    }

    ###############################
    #        Data loading         #
    ###############################
    (dataset, _, dates_start, dates_test,) = create_custom_datasets(file_path='train.xlsx', forecast_horizon=6, context_lenght=nn_config["context_length"], technical_feats=True, split_ratio = 0.20, lags=None)
    input_train = dataset.X[0:300]
    input_val = dataset.X[300:]
    target_train = dataset.y[0:300]
    target_val = dataset.y[300:]
    # # 1. Load the timeseries data.
    # (input_train, input_val, target_train, target_val,) = load_timeseries(train_dir, val_dir,)
    # 2. Convert the data to tensors and create input - target pairs. 
    (train_dataset, val_dataset,) = create_input_target_pairs(input_train, target_train, input_val, target_val,)
    # 3. Create data loaders.
    (train_loader, val_loader,) = data_loader(train_dataset, val_dataset, nn_config['batch_size'],)

    ###############################
    #      NN model creation      #
    ###############################
    # 1. Call the created nn model
    network = dense_model() #TODO
    network.to(device)
    print(network)
    # 2. Define the loss function (Criterion) and the optimizer.
    criterion = nn.MSELoss()
    criterion.to(device)
    optimizer = optim.Adam(network.parameters(), lr=nn_config['learning_rate'],) #TODO Can try other optimizers
    
    ###############################
    #   NN model Training & Val   #
    ###############################
    epoch_train_losses = []
    epoch_val_losses = []
    best_valid_loss = float('inf') # If the loss function need to be minimized, else -inf.
    best_train_loss = float('inf') # Ibid.

    # Itterate over the epochs.
    for epoch in range(1, nn_config['num_epochs']):
        print(f"Epoch {epoch+1}/{nn_config['num_epochs']}")

        #########################
        # 1. Training phase
        #########################
        network.train()
        train_loss = 0.0

        for (inputs, targets) in tqdm(train_loader, desc=f"Epoch {epoch} / {nn_config['num_epochs']}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + Backward + Optimizer
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # Adjusting the training epoch loss
            train_loss += loss.item()
        # Print statistics.
        train_loss /= len(train_loader)
        epoch_train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")

        #########################
        # 1. Valid phase
        #########################
        network.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for(inputs, targets) in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Froward
                outputs = network(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        # Print statistics.
        valid_loss /= len(val_loader)
        epoch_val_losses.append(valid_loss)
        print(f"Validation Loss: {valid_loss:.4f}")

        # Save the best model.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_train_loss = train_loss
            torch.save(network.state_dict(), f"{results_dir}/best_model.pth")
            print(f"Best model saved with validation loss: {best_valid_loss:.4f}")

    ###############################
    # Plotting model performances #
    ###############################
    plt.plot(epoch_train_losses, label="Train loss")
    plt.plot(epoch_val_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Best validation loss: {best_valid_loss:.4f}, and associate best training loss {best_train_loss:.4f}")
    plt.legend()
    plt.savefig(f"{results_dir}/training_plot.png")

    print(f"The training took: {time.time() - timer} seconds.")