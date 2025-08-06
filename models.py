# src/models.py

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# --- Deep Learning Model ---

class SimpleCNN(nn.Module):
    """
    A simple CNN to process sequences of SST maps.
    It treats the 12 months of input as 12 channels of a single image.
    """
    def __init__(self, input_channels=12, forecast_horizon=36):
        super(SimpleCNN, self).__init__()
        
        # --- Convolutional Layers ---
        # These layers learn spatial features from the SST maps.
        
        # Input: (Batch, 12, Lat, Lon) -> Output: (Batch, 32, Lat, Lon)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=1, padding='same')
        self.relu1 = nn.ReLU()
        # Halves the spatial dimensions (Lat, Lon)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Input: (Batch, 32, Lat/2, Lon/2) -> Output: (Batch, 64, Lat/2, Lon/2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.relu2 = nn.ReLU()
        # Halves the spatial dimensions again
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Fully Connected Layers ---
        # These layers perform the final regression task.
        
        self.flatten = nn.Flatten()
        
        # To make this robust, we will calculate the flattened size dynamically
        # instead of hard-coding it. We can do this with a dummy input.
        # The lat/lon dimensions of the COBE-SST data are 88x180.
        dummy_input = torch.randn(1, input_channels, 180, 360)
        dummy_output = self.pool2(self.conv2(self.pool1(self.conv1(dummy_input))))
        flattened_size = dummy_output.view(1, -1).size(1)
        
        # This layer takes the flattened features and maps them to an intermediate space
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu3 = nn.ReLU()
        
        # The output layer must have `forecast_horizon` neurons
        self.fc2 = nn.Linear(512, forecast_horizon)

    def forward(self, x):
        """The forward pass of the model."""
        # Input x is expected to have shape (Batch, 12, Lat, Lon)
        
        # Pass through convolutional blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Traditional Models ---

def get_ridge_model(alpha=1.0):
    """
    Returns a Ridge model wrapped in a MultiOutputRegressor.
    This allows a single scikit-learn model to predict multiple targets.
    """
    model = Ridge(alpha=alpha)
    # MultiOutputRegressor trains one separate regressor for each of the 36 target months.
    return MultiOutputRegressor(model)

def get_random_forest_model(n_estimators=100, max_depth=10, random_state=42):
    """
    Returns a Random Forest model wrapped in a MultiOutputRegressor.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        n_jobs=-1, # Use all available CPU cores
        random_state=random_state
    )
    return MultiOutputRegressor(model)