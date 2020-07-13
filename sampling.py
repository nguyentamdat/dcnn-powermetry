import numpy as np
from itertools import *
from functools import *
import multiprocessing as mp
import math
import h5py
from sklearn.model_selection import train_test_split


def load_data(filename):
    return np.load(filename, allow_pickle=True)


def sampling(data, num):
    try:
        data, y = data
        if math.isnan(y):
            return []
        last_point = np.argmax(data[:, 0])
        res = [(np.concatenate([data[sorted(np.random.choice(last_point, 24, replace=False))
                                     ], [data[last_point]]]), y) for _ in range(num)]
        return res
    except:
        return []


def sampling_all(data, num):
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(sampling, zip(data, repeat(num)))
    res = reduce(lambda x, y: x + y, res, [])
    return np.array(res)


if __name__ == "__main__":
    data = load_data("data/batt_filter.npy")
    # print(data[2][:, 1])
    # print(data)
    # a = sampling_all(data[0], 500)
    a = []
    for x in data:
        a.append(sampling_all(x, 5000))
    a = np.array(a)
    print(a.shape, a[0].shape, a[0][0].shape)
    np.savez_compressed("data/batt_sam_fil.npz", x=a)
    # print(a)
    # data = sampling_all(data, 5000)
    # print(data.shape)
    # print(math.isnan(data[-1][-1]))
    # x = np.asarray(data[:, 0])
    # y = np.asarray(data[:, 1])
    # x = np.array(list(map(lambda z: np.array(z), x)))
    # y = np.array(list(map(lambda z: np.array(z), y)))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # np.savez_compressed("data/batt_processed.npz", x_train=x_train,
    #                     x_test=x_test, y_train=y_train, y_test=y_test)
