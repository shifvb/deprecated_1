import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

workspace = r"F:\toTransfer\配准效果备份\2018_04_17_08_35_实验一_(将reg.v加入loss，且可训练)_imgnum=60410_imgsize=128x128_batch=80_iter=10000\temp_v_weights"
filenames = os.listdir(workspace)
filenames = [os.path.join(workspace, _) for _ in filenames]

arrs = []
for filename in filenames:
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
        for batch in range(arr.shape[0]):
            arr_c0 = arr[batch, :, :, 0]
            arr_c1 = arr[batch, :, :, 1]
            arrs.append(arr)

min_list = []
max_list = []
avg_list = []
var_list = []
for arr in arrs:
    min_list.append(arr.min())
    max_list.append(arr.max())
    avg_list.append(arr.mean())
    var_list.append(arr.var())

# plt.plot(range(1000), min_list)
# plt.plot(range(1000), max_list)
# plt.plot(range(1000), avg_list)
plt.plot(range(len(var_list)), var_list)
plt.show()

