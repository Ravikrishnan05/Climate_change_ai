# src/train.py

import argparse
import xarray as xr
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

# Import our custom modules
from data_loader import SSTDataset, get_traditional_ml_data_with_pca, load_enso_indices
from models import SimpleCNN, get_ridge_model, get_random_forest_model

# --- CHRONOLOGICAL SPLIT DEFINITION ---
# We define our training and validation periods here. This is crucial for
# the "strict chronological cross-validation" you mentioned in your resume.
# We leave a gap between train/val and val/test to prevent data leakage.
TRAIN_START, TRAIN_END = '1982-01-01', '2005-12-31'
# A validation set is used to tune hyperparameters, but for simplicity, we'll
# just use it to monitor training. We won't implement early stopping here.
VALID_START, VALID_END = '2006-01-01', '2010-12-31'


def train_cnn(args):
    """Handles the entire training and validation process for the CNN."""
    print("--- Training CNN ---")
    
    # 1. Load data
    print("Loading data...")
    sst_data = xr.open_dataset(args.data_path)
    enso_series = load_enso_indices()
    
    train_dataset = SSTDataset(sst_data, enso_series, TRAIN_START, TRAIN_END)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    valid_dataset = SSTDataset(sst_data, enso_series, VALID_START, VALID_END)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error is a good choice for regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # Use tqdm for a nice progress bar
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation loop
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad(): # We don't need to calculate gradients for validation
            for inputs, targets in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    # 4. Save the trained model
    # In train_cnn function
    save_path = os.path.join(PROJECT_ROOT, 'saved_models', f"cnn_model_epochs_{args.epochs}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"CNN model saved to {save_path}")

# IN src/train.py, REPLACE THE train_sklearn FUNCTION

def train_sklearn(model_name, args):
    """Handles the training for scikit-learn models using PCA."""
    print(f"--- Training {model_name.upper()} with PCA---")
    
    # 1. Load and prepare data using the new PCA function
    sst_data = xr.open_dataset(args.data_path)
    enso_series = load_enso_indices()
    # We only need the training data here, so we use underscores for the test data
    X_train_pca, y_train, _, _, pca_model = get_traditional_ml_data_with_pca(
        sst_data, enso_series, TRAIN_START, TRAIN_END, VALID_START, VALID_END, n_components=50
    )

    # 2. Initialize model
    if model_name == 'ridge':
        model = get_ridge_model(alpha=1.0) # Alpha might need tuning with PCA
    elif model_name == 'rf':
        # RF is much faster on 50 features than on 777k!
        model = get_random_forest_model(n_estimators=100, max_depth=10)
    
    # 3. Train model
    print(f"Fitting {model_name} model on PCA-transformed data...")
    model.fit(X_train_pca, y_train)
    
    # 4. Save the trained model and the PCA model (we need it for evaluation)
    model_save_path = os.path.join(PROJECT_ROOT, 'saved_models', f"{model_name}_model_pca.joblib")
    joblib.dump(model, model_save_path)
    print(f"{model_name} model saved to {model_save_path}")

    pca_save_path = os.path.join(PROJECT_ROOT, 'saved_models', 'pca_model.joblib')
    joblib.dump(pca_model, pca_save_path)
    print(f"PCA model saved to {pca_save_path}")

# ADD THIS AT THE TOP OF THE FILE
import os

# GET THE ROOT DIRECTORY OF THE PROJECT (same logic as before)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    # This block allows us to run the script from the command line
    parser = argparse.ArgumentParser(description="Train a model for ENSO forecasting.")
    parser.add_argument('model', type=str, choices=['cnn', 'ridge', 'rf'], 
                        help='The model to train: "cnn", "ridge", or "rf".')
    
    # Add arguments specific to the CNN
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for CNN.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs for CNN.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for CNN training.')
    
    # Add a general argument for the data path, now with a robust absolute path
    default_data_path = os.path.join(PROJECT_ROOT, 'data', 'sst.mon.mean.trefadj.anom.1880to2018.nc')
    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help='Path to the SST NetCDF data file.')

    args = parser.parse_args()

    if args.model == 'cnn':
        train_cnn(args)
    else:
        train_sklearn(args.model, args)