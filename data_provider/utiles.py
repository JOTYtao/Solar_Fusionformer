from functools import reduce
from typing import Any, Tuple
import numpy as np


def calculate_possible_starts(*dates: Any, frames_total: int) -> Tuple[Any]:
    """
    Computes the intersection of all input dates and outputs the indices where each index has the next ``frames_total`` timesteps available
    Adapted from: https://github.com/holmdk/IrradianceNet/blob/main/src/data/process_raw_data.py#L56
    Args:
        *dates, Any: datetime arrays or list of datetime.
        frames_total, int: Total number of timesteps to be available.
    Returns:
        possible_indices_dates, Any: Indices for each date in ``dates`` where each index has the next ``frames_total`` timesteps available.
    """
    assert len(dates) >= 1, "You have to provide at least one array!"

    # First compute the dates intersection and get their indices

    date_intersection = reduce(np.intersect1d, dates)
    intersection_indices = [
        np.where(np.isin(date, date_intersection))[0] for date in dates
    ]

    difference_range = np.diff(date_intersection)

    counted = np.zeros(difference_range.shape)
    for idx, time in enumerate(difference_range):
        if idx != counted.shape[0] - 1:
            if time == np.timedelta64(300, "s"):
            # if time == np.timedelta64(60000000000, "ns"):
                counted[idx + 1] = 1

    cum_sum = counted.copy()

    for idx, time in enumerate(counted):
        if idx > 0:
            if counted[idx] > 0:
                cum_sum[idx] = cum_sum[idx - 1] + cum_sum[idx]

    possible_indices = np.array(
        np.where(cum_sum >= (frames_total - 1))
    ).ravel()  # 1 since it is index

    # we use the beginning of the sequence as index
    possible_starts = possible_indices - (frames_total - 1)
    possible_starts = possible_starts.astype("int")

    possible_starts.sort()

    # Return possible indices from the original dates
    possible_indices_dates = tuple(
        [
            intersection_date[possible_starts]
            for intersection_date in intersection_indices
        ]
    )
    if len(possible_indices_dates) == 1:
        return possible_indices_dates[0]
    return possible_indices_dates


class MyTensor:
    timeseries_channels = ['GHI', 'RH [%]', 'T2 [°C]', 'PoPoPoPo [hPa]', 'DIF [W/m**2]', 'DIR [W/m**2]', 'ghi', 'dni',
                           'dhi']
    context_channels = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008',
                        'WV_062', 'WV_073']
    optflow_channels = ['IR_016_vx', 'IR_016_vy', 'IR_039_vx', 'IR_039_vy', 'IR_087_vx', 'IR_087_vy', 'IR_097_vx',
                        'IR_097_vy', 'IR_108_vx', 'IR_108_vy', 'IR_120_vx', 'IR_120_vy', 'IR_134_vx', 'IR_134_vy',
                        'VIS006_vx',
                        'VIS006_vy', 'VIS008_vx', 'VIS008_vy', 'WV_062_vx', 'WV_062_vy', 'WV_073_vx', 'WV_073_vy']

    def __init__(self, tensor):
        self.tensor = tensor
        self.info = {
            "timeseries_channels": MyTensor.timeseries_channels,
            "context_channels": MyTensor.context_channels,
            "optflow_channels": MyTensor.optflow_channels,

        }

    def get_ele_coord(self, station):
        if station == "PCCI_20082022_CAB":
            elevation = 0
            coordinates = [51.968, 4.928]
        elif station == "PCCI_20082022_CNR":
            elevation = 471
            coordinates = [42.816, 1.601]
        elif station == "PCCI_20082022_IZA":
            elevation = 2373
            coordinates = [28.3, 16.5]
        elif station == "PCCI_20082022_IZA_bis":
            elevation = 2373
            coordinates = [28.3, 16.5]
        elif station == "PCCI_20082022_PAL":
            elevation = 156
            coordinates = [48.713, 2.208]
        elif station == "PCCI_20082022_PAY":
            elevation = 491
            coordinates = [46.8123, 6.9422]
        elif station == "PCCI_20082022_TAM":
            elevation = 1385
            coordinates = [22.7903, 5.5292]
        else:
            elevation = None
            coordinates = None
        return elevation, coordinates

    def __getitem__(self, index):
        return self.tensor[index]

