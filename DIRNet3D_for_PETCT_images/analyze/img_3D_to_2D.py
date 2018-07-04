import os
import pickle
import numpy as np
from PIL import Image


def _trans(in_path, out_folder):
    # handle files % folders
    _name = os.path.split(in_path)[-1].split(".")[0]
    out_folder = os.path.join(out_folder, _name)
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    # extract data
    data = pickle.load(open(in_path, 'rb'))
    for i in range(data.shape[3]):
        _img_arr = (data[0, :, :, i, 0] * 255).astype(np.uint8)
        _img = Image.fromarray(_img_arr, "L")
        _img.save(os.path.join(out_folder, "{:>3}.jpg".format(i)))


def main():
    _trans(r"F:\KHJ\3D volume\pt_volume\03807.pkl", r"F:\temp_images")


if __name__ == '__main__':
    main()
