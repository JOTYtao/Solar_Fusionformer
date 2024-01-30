import requests
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

data_path = r"F:\eumetsat\2021\process\2021_nonhrv.zarr"
dataset = xr.open_zarr(data_path)

latitudes = dataset.latitude.values
longitudes = dataset.longitude.values
elevation_data = np.zeros((len(latitudes), len(longitudes)))
api_endpoint = "https://api.opentopodata.org/v1/test-dataset"
total_iterations = len(latitudes) * len(longitudes)
with tqdm(total=total_iterations, desc="Processing coordinates") as pbar:
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            params = {
                'locations': f"{lat},{lon}"
            }
            response = requests.get(api_endpoint, params=params)

            if response.status_code == 200:
                result = response.json()
                elevation_data[i, j] = result['results'][0]['elevation']
            else:
                print(f"Error: {response.status_code}")
                elevation_data[i, j] = np.nan

            pbar.update(1)

elevation_df = pd.DataFrame(elevation_data, index=latitudes, columns=longitudes)
elevation_array = elevation_df.to_numpy()
npy_path = r'F:\eumetsat\elevation\elevation_data.npy'
np.save(npy_path, elevation_array)
print("Elevation data saved to:", npy_path)