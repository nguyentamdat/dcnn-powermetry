import multiprocessing as mp
import pandas as pd
import glob
from itertools import *
import os
import numpy as np


def read_xlsx(filename, sheet=1):
    return pd.read_excel(filename, sheet_name=sheet)


def read_from_file():
    list_dir = glob.glob(os.curdir + "/data/stress*.xlsx")
    with mp.Pool(mp.cpu_count()) as p:
        data = p.starmap(read_xlsx, zip(list_dir, repeat(1)))
    return data


def get_xy(onecycle):
    return (onecycle[(onecycle.Step_Index == 1)][["Voltage(V)", "Current(A)", "Aux_Temperature(℃)_1"]].to_numpy(),
            onecycle[(onecycle.Step_Index == 3)]["Discharge_Capacity(Ah)"].max())


def get_data(data):
    res = []
    for datum in data:
        for cycle in np.unique(datum.Cycle_Index.to_numpy()):
            res.append(get_xy(datum[(datum.Cycle_Index == cycle)]))
    return np.array(res)


def get_data_pf(data):
    res = []
    for datum in data:
        item = []
        for cycle in np.unique(datum.Cycle_Index.to_numpy()):
            item.append(datum[(datum.Cycle_Index == cycle)][(
                datum.Step_Index == 3)]['Discharge_Capacity(Ah)'].max())
        res.append(np.array(item))
    return np.array(res)


if __name__ == "__main__":
    # a = get_data(read_from_file())
    # np.save("data/batt.npy", a)
    a = get_data_pf(read_from_file())
    np.save("data/batt_pf.npy", a)
