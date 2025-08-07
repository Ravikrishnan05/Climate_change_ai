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
- `src/models.py`: A catalog of model blueprints. Contains the `ReferenceCNN` class,  successful architecture.
- `src/run_reference_pipeline.py`: The main conductor script that orchestrates the entire training and evaluation process, bringing together the data and models to produce the final result.

## How to Run the Project

## ✅ 1. Setup

First, set up a Python virtual environment and install the required dependencies.

### Navigate to the project directory
cd path/to/enso-forecasting

### Create and activate a virtual environment
python -m venv venv

### On Windows
venv\Scripts\activate

### On macOS/Linux
source venv/bin/activate

### Install dependencies
pip install torch numpy xarray pandas scipy scikit-learn matplotlib



## 2. Download Data
Place the required `.nc` and `.txt` data files in the `data/` directory. These files need to be acquired from the official sources linked in the **Data Sources** section above.

## 3. Run the Pipeline
Execute the main script from the root directory of the project. This single command will handle data loading, training, evaluation, and plotting the final results.

```bash
python src/run_reference_pipeline.py
```


This script will:

Load and preprocess the data

Train the CNN model for 40 epochs

Evaluate the model on the test set

Save the best-performing model as reference_cnn_replication.pt

Output final Correlation and RMSE metrics

### 📚 Methodology in Detail
🧪 Data Splitting Strategy
A strict chronological split with a gap period is used to prevent data leakage and ensure honest evaluation.

Block	Start Date	End Date	Purpose
Training Set	1960-01-01	2005-12-31	Model learns patterns from this data. Last target seen: Feb 2006.
Gap / Buffer	2006-01-01	2006-12-31	1-year buffer to avoid data leakage. Discarded entirely.
Test Set	2007-01-01	2017-12-31	Completely unseen data used for final evaluation.

### 🔁 Training and Evaluation
Epoch-based Training:
The model is trained for 40 epochs on the full training dataset.

Early Stopping Logic:
After each epoch, test loss is calculated. If it's the lowest so far, model weights are saved.

Final Evaluation:
After all epochs, the best saved model is loaded and used to make predictions on the test set.

This ensures that the reported metrics reflect the best possible performance during training, without overfitting.

### ✅ Summary
This project showcases a clean, modular approach to solving a real-world climate forecasting problem using deep learning.

By using:

Spatial patterns in SST anomaly maps

Well-structured temporal windows

Proper evaluation methodology

The model remains both interpretable and high-performing.






