import multiprocessing as mp
import pandas as pd
import glob
from itertools import *
import os
import numpy as np
import math


def read_xlsx(filename, sheet=1):
    return pd.read_excel(filename, sheet_name=sheet)


def read_from_file():
    list_dir = glob.glob(os.curdir + "/data/stress*.xlsx")
    with mp.Pool(mp.cpu_count()) as p:
        data = p.starmap(read_xlsx, zip(list_dir, repeat(1)))
    return data


def get_xy(onecycle):
    mm = onecycle[(onecycle.Step_Index == 3)]["Discharge_Capacity(Ah)"].max()
    return (onecycle[(onecycle.Step_Index == 1)][["Voltage(V)", "Current(A)", "Aux_Temperature(℃)_1"]].to_numpy(),
            np.array(mm)) if math.isnan(mm) is False else None


def get_data(data):
    res = []
    for datum in data:
        cell = []
        for cycle in np.unique(datum.Cycle_Index.to_numpy()):
            a = get_xy(datum[(datum.Cycle_Index == cycle)])
            if a:
                cell.append(np.array(a))
        res.append(np.array(cell))
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
    a = np.load("data/batt.npy", allow_pickle=True)
    # a = np.array([x for x in a if x.shape[0] != 0])
    print(a[0][0][1])
    b = []
    for x in a:
        c = [np.array(x[0])]
        for y in x[1:]:
            if abs(c[-1][1] - y[1]) < 0.2:
                c.append(np.array(y))
        b.append(np.array(c))
    b = np.array(b)
    print(b[0][0][0].shape)
    np.save("data/batt_filter.npy", b)
    # np.save("data/batt.npy", a)
    # a = get_data_pf(read_from_file())
    # np.save("data/batt_pf.npy", a)
