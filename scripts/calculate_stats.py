import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import json

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import pandas as pd
import os

os.environ["MY_UNIQUE_ID"] = "joty"
os.environ["HYDRA_FULL_ERROR"] = "1"

class MyTensor:
    def __init__(self, tensor, info):
        self.tensor = tensor
        self.info = info


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="calc_stats.yaml"
)
def main(cfg: DictConfig) -> None:
    dataset_path = cfg.dataset_path
    stations = list(map(str, cfg.station))
    print(stations)
    station_vars = {}
    station_channels = {}
    station_data = {}
    for station in stations:
        station_data[station] = []
        station_path = os.path.join(dataset_path, station)
        for year in os.listdir(station_path):
            data_path = os.path.join(station_path, year, "solar_data.csv")
            data = pd.read_csv(data_path)
            channels = ['direct_n', 'diffuse','temp', 'rh', 'windspd', 'pressure', 'CSI']
            ts_tensor = torch.from_numpy(data[channels].values)
            station_channels[station] = channels
            if not station in station_vars:
                station_vars[station] = [ts_tensor.numpy()]
            else:
                station_vars[station].append(ts_tensor.numpy())

    stats_dict = {}
    for station in station_vars:
        stats_dict[station] = {}
        x = np.concatenate(station_vars[station], axis=0)
        print(np.isnan(x).any())

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        print(mean)
        for i, chan in enumerate(station_channels[station]):
            stats_dict[station][chan] = {}
            stats_dict[station][chan]["mean"] = str(mean[i])
            stats_dict[station][chan]["std"] = str(std[i])


    with open(cfg.save_path, "w") as fp:
        json.dump(stats_dict, fp)
if __name__ == "__main__":
    main()