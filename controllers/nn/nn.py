import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class StrikerDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV
        self.data = pd.read_csv(csv_file)
        
        # Split into Inputs (X) and Outputs (Y)
        # First 15 columns are the observation space
        self.X = self.data.iloc[:, :15].values.astype(np.float32)
        
        # Last 3 columns are the action space (Left Motor, Right Motor, Shoot Flag)
        self.Y = self.data.iloc[:, 15:].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class BC_Policy(nn.Module):
    def __init__(self):
        super(BC_Policy, self).__init__()
        
        # 15 Inputs -> 64 -> 64 -> 3 Outputs
        self.network = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # No activation at the end because motor speeds can be negative
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    csv_path = "C:\\Users\\dell\\Desktop\\robotics\\controllers\\Dataset_gen\\expert_demonstrations.csv"
    
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Go play the game first!")
        return

    # Load data
    print("Loading dataset...")
    dataset = StrikerDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"Loaded {len(dataset)} expert frames.")

    # Initialize Network, Loss Function, and Optimizer
    model = BC_Policy()
    criterion = nn.MSELoss() # Mean Squared Error works great for motor speed regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50 # Start small just to test the pipeline

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_X, batch_Y in dataloader:
            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass (Predict actions)
            predictions = model(batch_X)
            
            # 3. Calculate how wrong the predictions were
            loss = criterion(predictions, batch_Y)
            
            # 4. Backward pass (Calculate gradients)
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print progress
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    # Save the trained brain
    torch.save(model.state_dict(), "bc_policy.pth")
    print("\nTraining Complete! Brain saved to 'bc_policy.pth'")

if __name__ == "__main__":
    train()