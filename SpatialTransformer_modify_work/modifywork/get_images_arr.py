import os
from functools import reduce
import numpy as np
from PIL import Image


def process_images(load_dir: str):
    abs_img_filenames = [os.path.join(load_dir, _) for _ in os.listdir(load_dir)]
    img_arrs = []
    for abs_img_filename in abs_img_filenames:
        img = Image.open(abs_img_filename)
        arr = np.array(img)
        assert len(arr.shape) == 3
        assert arr.shape[2] == 3
        print("in: ", abs_img_filename, arr.dtype, arr.shape)
        if arr.shape[1] < arr.shape[0]:
            arr = arr.transpose([1, 0, 2])
        if abs(arr.shape[1] / arr.shape[0] - 1920 / 1080) > 1e-2:
            continue
        img_arrs.append(arr)
    img_arrs.sort(key=lambda _: reduce(lambda _x, _y: _x * _y, _.shape), reverse=True)
    for i, arr in enumerate(img_arrs):
        img = Image.fromarray(arr)
        img = img.resize([384, 216], resample=Image.BICUBIC)  # row=1080, column=1920
        img.save(os.path.join(r"f:\tmp2", "{:>03}.png".format(i)))


def get_images_arr(load_dir: str):
    abs_img_filenames = [os.path.join(load_dir, _) for _ in os.listdir(load_dir)]
    return np.stack([Image.open(_) for _ in abs_img_filenames], axis=0)


if __name__ == '__main__':
    # process_images(r"F:\tmp")
    get_images_arr(r"f:\tmp2")
