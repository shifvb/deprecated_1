import os
from PIL import Image
import numpy as np


def cal_diff(_a, _b):
    return np.abs(np.array(_a, dtype=np.int32) - _b.astype(np.int32))


def main():
    workspace = r"F:\registration_running_data\validate"
    out_dir = r"F:\registration_running_data\diff"
    images = [os.path.join(workspace, _) for _ in os.listdir(workspace)]
    for i in range(0, len(images), 3):
        _name = os.path.split(images[i])[1].split("_")[0]
        _name = str(_name)
        arr_x = np.array(Image.open(images[i]))
        arr_y = np.array(Image.open(images[i + 1]))
        arr_z = np.array(Image.open(images[i + 2]))
        arr_12 = cal_diff(arr_x, arr_y)
        arr_23 = cal_diff(arr_y, arr_z)
        arr_12 = arr_12.astype(np.uint8)
        arr_23 = arr_23.astype(np.uint8)
        Image.fromarray(arr_12).save(os.path.join(out_dir, _name + "_x1-x2.png"))
        Image.fromarray(arr_23).save(os.path.join(out_dir, _name + "_x2-x3.png"))
        # break


if __name__ == '__main__':
    main()
