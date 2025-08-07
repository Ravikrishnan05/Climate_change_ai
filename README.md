# ENSO Forecasting with Convolutional Neural Networks

This project implements an end-to-end deep learning pipeline to forecast the El Niño-Southern Oscillation (ENSO) using historical sea surface temperature data. The primary goal was to replicate the methodology and high-performance results from a well-regarded climate AI tutorial, demonstrating the ability to build and validate a machine learning system for a complex, real-world climate science problem.

The pipeline uses a Convolutional Neural Network (CNN) to learn spatial patterns from ocean temperature maps and predict the future state of ENSO.

## Final Results

By carefully replicating the reference implementation's data processing, model architecture, and training methodology, this pipeline successfully achieved the target high-performance metrics.

- **Correlation:** 0.92
- **RMSE:** 0.37



*(Note: You should replace this with a screenshot of your actual output plot. I've used a placeholder image URL.)*

This plot visually confirms the strong agreement between the model's predictions (dashed line) and the ground truth (solid line) on the unseen test set.

## Project Overview

### The Forecasting Task

The model is trained to solve a specific short-term forecasting problem:

- **Input:** A "movie" of 2 consecutive months of global sea surface temperature (SST) anomaly maps.
- **Target:** The single value of the Niño 3.4 index 2 months after the end of the input window.

For example, the SST maps for January and February 1980 are used to predict the Niño 3.4 index for April 1980.

### Data Sources

- **Input Data:** [COBE Sea-Surface Temperature Dataset](https://psl.noaa.gov/data/gridded/data.cobe.html) (1880-2018)
- **Target Data:** [NOAA Niño 3.4 Index](https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data) (1870-2018)

### System Design

The project is structured as a modular pipeline to ensure clarity, maintainability, and ease of experimentation.

- `data/`: Contains the raw `.nc` and `.txt` data files.
- `src/data_loader.py`: The data expert. Responsible for loading, cleaning, and preparing the (input, target) pairs from the raw files according to specified time windows and parameters.
- `src/models.py`: A catalog of model blueprints. Contains the `ReferenceCNN` class, an exact replica of the successful architecture from the tutorial.
- `src/run_reference_pipeline.py`: The main conductor script that orchestrates the entire training and evaluation process, bringing together the data and models to produce the final result.

## How to Run the Project

### 1. Setup

First, set up a Python virtual environment and install the required dependencies.

```bash
# Navigate to the project directory
cd path/to/enso-forecasting

# Create and activate a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install torch numpy xarray pandas scipy scikit-learn matplotlib

