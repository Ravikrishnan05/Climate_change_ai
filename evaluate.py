# src/evaluate.py

import xarray as xr
import torch
import joblib
import numpy as np
import os
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd 


# Import our custom modules
from data_loader import SSTDataset, get_traditional_ml_data, load_enso_indices
from models import SimpleCNN, get_ridge_model, get_random_forest_model

# Define the Test Set period. This data has NEVER been seen by any model.
TEST_START, TEST_END = '2011-01-01', '2017-12-31'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_models(args):
    """Loads all trained models and evaluates them on the test set."""
    print("--- Model Evaluation on Test Set ---")
    
    # 1. Load Data and Models
    print("Loading data and models...")
    sst_data = xr.open_dataset(os.path.join(PROJECT_ROOT, 'data', 'sst.mon.mean.trefadj.anom.1880to2018.nc'))
    enso_series = load_enso_indices()
    
    # Load CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = SimpleCNN().to(device)
    cnn_model_path = os.path.join(PROJECT_ROOT, 'saved_models', f'cnn_model_epochs_{args.epochs}.pth')
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()
    
    # Load Sklearn models (trained on PCA data)
    ridge_model = joblib.load(os.path.join(PROJECT_ROOT, 'saved_models', 'ridge_model_pca.joblib'))
    rf_model = joblib.load(os.path.join(PROJECT_ROOT, 'saved_models', 'rf_model_pca.joblib'))
    pca_model = joblib.load(os.path.join(PROJECT_ROOT, 'saved_models', 'pca_model.joblib'))
    
    # 2. Prepare Test Data
    print(f"Preparing test data for the period: {TEST_START} to {TEST_END}")
    # For CNN
    cnn_test_dataset = SSTDataset(sst_data, enso_series, TEST_START, TEST_END)
    # For sklearn models, we get the raw flattened data first
    X_test_flat, y_test_true = get_traditional_ml_data(sst_data, enso_series, TEST_START, TEST_END)
    # Then we transform it with our loaded PCA model
    X_test_pca = pca_model.transform(X_test_flat)

    # 3. Generate Predictions
    print("Generating predictions...")
    # (The prediction generation code remains the same, but now uses X_test_pca for sklearn)
    cnn_preds = []
    with torch.no_grad():
        for i in tqdm(range(len(cnn_test_dataset)), desc="CNN Predicting"):
            input_tensor, _ = cnn_test_dataset[i]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            output = cnn_model(input_tensor)
            cnn_preds.append(output.squeeze().cpu().numpy())
    cnn_preds = np.array(cnn_preds)
    
    ridge_preds = ridge_model.predict(X_test_pca)
    rf_preds = rf_model.predict(X_test_pca)
    
    # 4. Calculate Metrics and Plots (This part remains exactly the same as before)
    # ... (copy the entire metrics and plotting section from the previous evaluate_models version) ...
    print("\n--- Performance Metrics ---")
    lead_times_to_evaluate = [0, 5, 11, 23, 35]
    
    for lead_idx in lead_times_to_evaluate:
        lead_month = lead_idx + 1
        print(f"\n--- Lead Time: {lead_month} months ---")
        
        true_vals = y_test_true[:, lead_idx]
        cnn_vals = cnn_preds[:, lead_idx]
        ridge_vals = ridge_preds[:, lead_idx]
        rf_vals = rf_preds[:, lead_idx]
        
        # Correlation
        corr_cnn, _ = pearsonr(true_vals, cnn_vals)
        corr_ridge, _ = pearsonr(true_vals, ridge_vals)
        corr_rf, _ = pearsonr(true_vals, rf_vals)
        
        # RMSE
        rmse_cnn = np.sqrt(mean_squared_error(true_vals, cnn_vals))
        rmse_ridge = np.sqrt(mean_squared_error(true_vals, ridge_vals))
        rmse_rf = np.sqrt(mean_squared_error(true_vals, rf_vals))
        
        print(f"  Correlation:")
        print(f"    CNN: {corr_cnn:.3f}, Ridge: {corr_ridge:.3f}, RF: {corr_rf:.3f}")
        print(f"  RMSE:")
        print(f"    CNN: {rmse_cnn:.3f}, Ridge: {rmse_ridge:.3f}, RF: {rmse_rf:.3f}")

    # 5. Create a comparison plot for a specific lead time (e.g., 6 months)
    lead_to_plot = 5 # 6-month forecast
    plt.figure(figsize=(15, 7))
    plt.title(f'Model Comparison on Test Set (Lead Time: {lead_to_plot+1} months)', fontsize=16)
    
    # To get the date index for plotting
    test_start_date = pd.to_datetime(TEST_START)
    num_test_samples = len(y_test_true)
    # The date corresponds to the time of the *forecast*, not the input
    plot_dates = pd.date_range(start=test_start_date + pd.DateOffset(months=12+lead_to_plot), periods=num_test_samples, freq='MS')
    
    plt.plot(plot_dates, y_test_true[:, lead_to_plot], label='Ground Truth (Actual ENSO)', color='black', linewidth=2)
    plt.plot(plot_dates, cnn_preds[:, lead_to_plot], label=f'CNN (Corr: {pearsonr(y_test_true[:, lead_to_plot], cnn_preds[:, lead_to_plot])[0]:.2f})', linestyle='--')
    plt.plot(plot_dates, ridge_preds[:, lead_to_plot], label=f'Ridge (Corr: {pearsonr(y_test_true[:, lead_to_plot], ridge_preds[:, lead_to_plot])[0]:.2f})', linestyle='--')
    plt.plot(plot_dates, rf_preds[:, lead_to_plot], label=f'Random Forest (Corr: {pearsonr(y_test_true[:, lead_to_plot], rf_preds[:, lead_to_plot])[0]:.2f})', linestyle='--')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Nino3.4 Index Anomaly (K)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_save_path = os.path.join(PROJECT_ROOT, 'model_comparison_plot.png')
    plt.savefig(plot_save_path)
    print(f"\nComparison plot saved to: {plot_save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test set.")
    parser.add_argument('--epochs', type=int, default=15, help='Specify the number of epochs the CNN was trained for, to load the correct file.')
    args = parser.parse_args()
    evaluate_models(args)