import os
import pickle
import numpy as np
from PIL import Image


def _trans(in_path, out_folder, axis):
    # handle files % folders
    _name = os.path.split(in_path)[-1].split(".")[0]
    out_folder = os.path.join(out_folder, _name)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    # extract data
    data = pickle.load(open(in_path, 'rb'))
    if axis is not None:
        data = data.transpose(axis)
    print("[DEBUG] loaded {} with:\n\tshape: {}\n\tdtype:{}".format(in_path, data.shape, data.dtype))
    if data.ndim == 5:
        for i in range(data.shape[3]):
            _img_arr = (data[0, :, :, i, 0] * 255).astype(np.uint8)
            _img = Image.fromarray(_img_arr, "L")
            _img.save(os.path.join(out_folder, "{:>3}.jpg".format(i)))
    elif data.ndim == 3:
        for i in range(data.shape[2]):
            _img_arr = (data[:, :, i] * 255).astype(np.uint8)
            _img = Image.fromarray(_img_arr, "L")
            _img.save(os.path.join(out_folder, "{:>3}.jpg".format(i)))
    # save data


def trans(in_folder, out_folder, axis=None):
    img_names = [os.path.join(in_folder, _) for _ in os.listdir(in_folder)]
    for img_name in img_names:
        _trans(img_name, out_folder, axis)


def main():
    trans(r"F:\新建文件夹", r"d:\result_hori")
    trans(r"F:\新建文件夹", r"d:\result_sagi", [2, 0, 1])
    trans(r"F:\新建文件夹", r"d:\result_coro", [2, 1, 0])

if __name__ == '__main__':
    main()
