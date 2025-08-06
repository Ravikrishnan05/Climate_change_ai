# src/data_loader.py

import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA

# ADD THIS AT THE TOP OF THE FILE
import os

# GET THE ROOT DIRECTORY OF THE PROJECT
# __file__ is the path to the current script (data_loader.py)
# os.path.dirname(__file__) is the directory of the script ('.../src/')
# os.path.dirname(...) goes up one level to the project root ('.../enso-forecasting/')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_enso_indices():
    """
    Reads in the txt data file to output a pandas Series of ENSO vals.
    This is copied from the tutorial notebook.
    
    Outputs
    -------
        pd.Series : monthly ENSO values starting from 1870-01-01
    """
    # Construct the absolute path to the data file
    file_path = os.path.join(PROJECT_ROOT, 'data', 'nino34.long.anom.data.txt')
    with open(file_path) as f:
        line = f.readline()
        enso_vals = []
        while line:
            yearly_enso_vals = map(float, line.split()[1:])
            enso_vals.extend(yearly_enso_vals)
            line = f.readline()

    enso_vals = pd.Series(enso_vals)
    enso_vals.index = pd.date_range('1870-01-01',freq='MS',
                                  periods=len(enso_vals))
    enso_vals.index = pd.to_datetime(enso_vals.index)
    return enso_vals

class SSTDataset(Dataset):
    """
    PyTorch Dataset for loading SST data for the CNN.
    This version pre-aligns the data upon initialization for robustness.
    """
    def __init__(self, sst_data, enso_series, start_date, end_date, input_months=12, forecast_months=36):
        self.input_months = input_months
        self.forecast_months = forecast_months
        
        # --- Data Alignment ---
        # 1. Select the main data for the given time period
        sst_slice = sst_data['sst'].sel(time=slice(start_date, end_date))
        
        # 2. Reindex the ENSO series to match the exact timestamps of the SST data.
        # 'ffill' (forward fill) ensures that for a date like '1982-01-16', we get the ENSO value from '1982-01-01'.
        aligned_enso = enso_series.reindex(sst_slice.time, method='ffill')

        self.sst_values = sst_slice.values
        self.enso_values = aligned_enso.values
        
        # Replace any NaN values that might exist
        self.sst_values[np.isnan(self.sst_values)] = 0
        self.enso_values[np.isnan(self.enso_values)] = 0

    def __len__(self):
        """Returns the total number of samples we can create."""
        # The number of samples is the total time steps minus the windows needed for input and forecast.
        return len(self.sst_values) - self.input_months - self.forecast_months + 1

    def __getitem__(self, idx):
        """
        Retrieves one sample (input and target) from the dataset.
        This now works with simple numpy array indexing, which is fast and reliable.
        """
        # 1. Define the slice for the input data based on the index
        start_input_idx = idx
        end_input_idx = idx + self.input_months
        
        # 2. Get the input chunk from our pre-aligned numpy array
        input_array = self.sst_values[start_input_idx:end_input_idx]
        
        # 3. Define the slice for the target data
        # The target starts *after* the input data ends.
        start_target_idx = end_input_idx
        end_target_idx = start_target_idx + self.forecast_months
        
        # 4. Get the target chunk from our pre-aligned numpy array
        target_array = self.enso_values[start_target_idx:end_target_idx]

        # 5. Convert numpy arrays to PyTorch tensors
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        target_tensor = torch.tensor(target_array, dtype=torch.float32)
        
        return input_tensor, target_tensor

def get_traditional_ml_data(sst_data, enso_series, start_date, end_date, input_months=12, forecast_months=36):
    """
    Prepares flattened data for use with scikit-learn models (Ridge, RandomForest).
    It uses the SSTDataset internally and then flattens the input.
    """
    # Use our SSTDataset to generate the samples
    dataset = SSTDataset(sst_data, enso_series, start_date, end_date, input_months, forecast_months)
    
    X_list, y_list = [], []
    print(f"Generating flattened data for sklearn from {start_date} to {end_date}...")
    for i in range(len(dataset)):
        # Get the tensor data
        input_tensor, target_tensor = dataset[i]
        
        # Flatten the input tensor from (12, lat, lon) to a 1D vector and append
        X_list.append(input_tensor.numpy().flatten())
        # Append the target vector
        y_list.append(target_tensor.numpy())
        
    return np.array(X_list), np.array(y_list)

# ADD THIS NEW FUNCTION AT THE END OF data_loader.py

def get_traditional_ml_data_with_pca(sst_data, enso_series, train_start, train_end, test_start, test_end, n_components=50):
    """
    Prepares flattened data for sklearn models and applies PCA.
    PCA is fitted ONLY on the training data and then used to transform both train and test data.
    """
    print("--- Preparing data with PCA ---")
    # First, get the raw flattened data for both train and test periods
    X_train_raw, y_train = get_traditional_ml_data(sst_data, enso_series, train_start, train_end)
    X_test_raw, y_test = get_traditional_ml_data(sst_data, enso_series, test_start, test_end)
    
    # Initialize the PCA model
    print(f"Fitting PCA to training data and reducing to {n_components} components...")
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    
    # Fit the PCA model ONLY on the training data to learn the components
    pca.fit(X_train_raw)
    
    # Use the FITTED pca model to transform both the training and test data
    X_train_pca = pca.transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)
    
    print(f"Original feature size: {X_train_raw.shape[1]}")
    print(f"New feature size after PCA: {X_train_pca.shape[1]}")
    
    return X_train_pca, y_train, X_test_pca, y_test, pca