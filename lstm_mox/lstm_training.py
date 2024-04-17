# load data from catalog
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import time
import matplotlib.pyplot as plt
from typing import Dict
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import IntegratedGradients

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def sequence_generate(sequence_length: int, data_set: pd.DataFrame, features: list, target_column: str, step_size: int):
    # Initialize lists to hold sequences and targets
    sequences = []
    targets = []
    # Group by 'exp_no' and create sequences for each group
    for _, group in data_set.groupby('exp_no'):
        data = group[features].values
        target_data = group[target_column].values
        # Create sequences with specified step size
        for i in range(0, len(group) - sequence_length, step_size):
            # Extract the sequence of features and the corresponding target
            sequence = data[i:(i + sequence_length)]
            target = target_data[i + sequence_length]  # Target is the next record
            sequences.append(sequence)
            targets.append(target)
    # Convert lists to numpy arrays
    sequences_np = np.array(sequences)
    targets_np = np.array(targets)
    return sequences_np, targets_np

def split_data(sequences, targets, test_size=0.2, random_state=42):
    sequences_train, sequences_test, targets_train, targets_test = train_test_split(
        sequences, targets, test_size=test_size, random_state=random_state)
    return sequences_train, sequences_test, targets_train, targets_test


def scale_sequences(sequences_train, sequences_test):
    scaler = StandardScaler()
    n_samples_train, sequence_length, n_features = sequences_train.shape
    sequences_train_reshaped = sequences_train.reshape(-1, n_features)
    scaler.fit(sequences_train_reshaped)
    sequences_train_scaled = scaler.transform(sequences_train_reshaped).reshape(n_samples_train, sequence_length, n_features)
    
    n_samples_test, _, _ = sequences_test.shape
    sequences_test_reshaped = sequences_test.reshape(-1, n_features)
    sequences_test_scaled = scaler.transform(sequences_test_reshaped).reshape(n_samples_test, sequence_length, n_features)
    
    return sequences_train_scaled, sequences_test_scaled


def convert_to_tensors(sequences_train, sequences_test, targets_train, targets_test):
    train_sequences_tensor = torch.tensor(sequences_train, dtype=torch.float32)
    test_sequences_tensor = torch.tensor(sequences_test, dtype=torch.float32)
    train_targets_tensor = torch.tensor(targets_train, dtype=torch.float32)
    test_targets_tensor = torch.tensor(targets_test, dtype=torch.float32)
    return train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor


def create_dataloaders(train_sequences_tensor, train_targets_tensor, test_sequences_tensor, test_targets_tensor, batch_size=64):
    train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
    test_dataset = TensorDataset(test_sequences_tensor, test_targets_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_training(model, learning_rate, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    for sequences_batch, targets_batch in train_loader:
        sequences_batch = sequences_batch.to(device)
        targets_batch = targets_batch.to(device).unsqueeze(-1)
        
        outputs = model(sequences_batch)
        loss = criterion(outputs, targets_batch)
        epoch_losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_training_loss = sum(epoch_losses) / len(epoch_losses)
    return avg_training_loss

def validate(model, test_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    count = 0
    with torch.no_grad():
        for sequences_batch, targets_batch in test_loader:
            sequences_batch = sequences_batch.to(device)
            targets_batch = targets_batch.to(device).unsqueeze(-1)
            
            outputs = model(sequences_batch)
            loss = criterion(outputs, targets_batch)
            
            total_val_loss += loss.item()
            count += 1
    
    avg_val_loss = total_val_loss / count
    return avg_val_loss

import torch
import matplotlib.pyplot as plt

def save_checkpoint(epoch, model, optimizer, avg_training_loss, val_rmse, session_checkpoint_dir):
    checkpoint_path = os.path.join(session_checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': avg_training_loss,
        'validation_rmse': val_rmse,
    }, checkpoint_path)

def plot_metrics(training_losses, validation_rmses, epoch, session_checkpoint_dir):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), validation_rmses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE Over Time')
    
    plt.tight_layout()
    plt.savefig(os.path.join(session_checkpoint_dir, f'plots_epoch_{epoch+1}.png'))
    plt.close()


# Check if CUDA is available, if not, check for MPS (Metal Performance Shaders for Apple Silicon), otherwise use CPU

# load aligned_df from path
# file_path = "aligned_df.pq"
# data_set = pd.read_parquet(file_path)


def main():
    """ include a config file as json file with the session timestamp that includes hyperparameter settings 
    features, target_column, sequence_length, step_size, batch_size, hidden_size, num_layers, num_classes, learning_rate,
    criterion, optimizer, num_epochs, 
    """



    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    file_path = "../data/aligned_df_with_water_ratio.pq"
    data_set = pd.read_parquet(file_path)

    features = ['timestamp_bin', 'A1_Resistance', 'A1_Resistance_diff']
    # target_column = 'resistance_ratio'
    target_column = 'water_ratio'
    sequence_length = 100
    step_size = 1
    sequences_np, targets_np = sequence_generate(sequence_length, data_set, features, target_column, step_size)

    sequences_train, sequences_test, targets_train, targets_test = split_data(sequences_np, targets_np)
    sequences_train_scaled, sequences_test_scaled = scale_sequences(sequences_train, sequences_test)

    batch_size = 32
    train_sequences_tensor, test_sequences_tensor, train_targets_tensor, test_targets_tensor = convert_to_tensors(
        sequences_train_scaled, sequences_test_scaled, targets_train, targets_test)
    train_loader, test_loader = create_dataloaders(
        train_sequences_tensor, train_targets_tensor, test_sequences_tensor, test_targets_tensor, batch_size=batch_size)

    input_size = len(features)
    hidden_size = 32
    num_layers = 2
    num_classes = 1
    model = RNN(input_size, hidden_size, num_layers, num_classes)
    learning_rate = 0.0001
    criterion, optimizer = setup_training(model, learning_rate=learning_rate, device=device)

    num_epochs = 100
    checkpoint_dir = './checkpoints'
    ensure_dir(checkpoint_dir)
    session_timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_checkpoint_dir = os.path.join(checkpoint_dir, f'training_session_{session_timestamp}')
    ensure_dir(session_checkpoint_dir)

    training_losses = []
    validation_rmses = []

    for epoch in range(num_epochs):
        avg_training_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        training_losses.append(avg_training_loss)

        avg_val_loss = validate(model, test_loader, criterion, device)
        val_rmse = math.sqrt(avg_val_loss)
        validation_rmses.append(val_rmse)

        print(f'Epoch {epoch+1}, Training Loss: {avg_training_loss}, Validation RMSE: {val_rmse}')

        save_checkpoint(epoch, model, optimizer, avg_training_loss, val_rmse, session_checkpoint_dir)

        if (epoch + 1) % 5 == 0:
            plot_metrics(training_losses, validation_rmses, epoch, session_checkpoint_dir)


if __name__ == "__main__":
    main()
