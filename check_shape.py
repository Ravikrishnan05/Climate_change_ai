# check_shape.py
import xarray as xr
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(PROJECT_ROOT, 'data', 'sst.mon.mean.trefadj.anom.1880to2018.nc')

sst_data = xr.open_dataset(data_path)

# Print the full dataset to inspect it
print(sst_data)

# Print the shape of the 'sst' variable
print("\nShape of the 'sst' variable (time, lat, lon):")
print(sst_data['sst'].shape)