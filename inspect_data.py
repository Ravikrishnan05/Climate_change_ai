# src/inspect_data.py

import xarray as xr
import os
import pandas as pd

# We need to import our SSTDataset class to test it
from data_loader import SSTDataset, load_enso_indices

# --- Configuration ---
# You can change these to see how the splits work for different periods

# This is our training period, as defined in train.py
START_DATE = '1982-01-01'
END_DATE = '2005-12-31'

# Let's inspect the first 3 samples and the last 3 samples
SAMPLES_TO_INSPECT_FROM_START = 3
SAMPLES_TO_INSPECT_FROM_END = 3


def inspect_dataset_splits():
    """
    This function loads the data, creates a dataset instance,
    and prints the date ranges for specific input/output samples.
    """
    print("--- Data Timeline Inspector ---")
    
    # 1. Load the raw data
    print("Loading raw SST and ENSO data...")
    sst_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sst.mon.mean.trefadj.anom.1880to2018.nc')
    sst_data = xr.open_dataset(sst_data_path)
    enso_series = load_enso_indices()
    print("Data loaded.\n")

    # 2. Create an instance of our dataset for the specified period
    print(f"Initializing SSTDataset for the period: {START_DATE} to {END_DATE}")
    dataset = SSTDataset(sst_data, enso_series, START_DATE, END_DATE)
    
    total_samples = len(dataset)
    print(f"Total number of samples that can be created in this period: {total_samples}\n")

    # 3. Inspect the first few samples
    print("--- Inspecting First Few Samples ---")
    for i in range(SAMPLES_TO_INSPECT_FROM_START):
        print(f"\n[Sample Index: {i}]")
        
        # Input period
        start_input_idx = i
        end_input_idx = i + dataset.input_months
        input_dates = dataset.sst_values.time[start_input_idx:end_input_idx]
        
        # Target period
        start_target_idx = end_input_idx
        end_target_idx = start_target_idx + dataset.forecast_months
        # CORRECTED: Access the .index of the pandas Series
        target_dates = dataset.enso_values.index[start_target_idx:end_target_idx]

        print(f"  INPUT  (12 months of SST maps):")
        print(f"    -> From: {input_dates.min().dt.strftime('%Y-%m').item()}")
        print(f"    -> To:   {input_dates.max().dt.strftime('%Y-%m').item()}")
        
        print(f"  OUTPUT (36 months of ENSO values):")
        # CORRECTED: Use .strftime directly on the timestamp object
        print(f"    -> From: {target_dates.min().strftime('%Y-%m')}")
        print(f"    -> To:   {target_dates.max().strftime('%Y-%m')}")
        

    # 4. Inspect the last few samples
    print("\n\n--- Inspecting Last Few Samples ---")
    for i in range(total_samples - SAMPLES_TO_INSPECT_FROM_END, total_samples):
        print(f"\n[Sample Index: {i}]")
        
        # Input period
        start_input_idx = i
        end_input_idx = i + dataset.input_months
        input_dates = dataset.sst_values.time[start_input_idx:end_input_idx]
        
        # Target period
        start_target_idx = end_input_idx
        end_target_idx = start_target_idx + dataset.forecast_months
        # CORRECTED: Access the .index of the pandas Series
        target_dates = dataset.enso_values.index[start_target_idx:end_target_idx]

        print(f"  INPUT  (12 months of SST maps):")
        print(f"    -> From: {input_dates.min().dt.strftime('%Y-%m').item()}")
        print(f"    -> To:   {input_dates.max().dt.strftime('%Y-%m').item()}")
        
        print(f"  OUTPUT (36 months of ENSO values):")
        # CORRECTED: Use .strftime directly on the timestamp object
        print(f"    -> From: {target_dates.min().strftime('%Y-%m')}")
        print(f"    -> To:   {target_dates.max().strftime('%Y-%m')}")
        

if __name__ == "__main__":
    # To make this script work, we need to temporarily modify the SSTDataset class
    # to expose the internal numpy arrays with their time coordinates.
    
    # We modify the __init__ of the original class for this inspection script
    original_init = SSTDataset.__init__
    def new_init(self, sst_data, enso_series, start_date, end_date, input_months=12, forecast_months=36):
        self.input_months = input_months
        self.forecast_months = forecast_months
        sst_slice = sst_data['sst'].sel(time=slice(start_date, end_date))
        
        # This is the key change: we store the xarray DataArrays, not just the values
        self.sst_values = sst_slice
        self.enso_values = enso_series.reindex(sst_slice.time, method='ffill')

    SSTDataset.__init__ = new_init
    
    # We also need to import pandas for this to work
    import pandas as pd
    
    inspect_dataset_splits()
    
    # Restore the original method so we don't break train.py
    SSTDataset.__init__ = original_init