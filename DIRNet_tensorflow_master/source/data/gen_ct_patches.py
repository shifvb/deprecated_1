import os
import pydicom
import math
import random
import pickle
import numpy as np
import SimpleITK as sitk
from PIL import Image


def _gen_batches(arr: list, middle_length: int = 200):
    """
    gen batch_x , batch_y
    :param arr:
    :return: batch_x: list of [512, 512] array
            batch_y: list of [512, 512] array
    """
    assert len(arr) > middle_length
    batch_x, batch_y = [], []
    _ = int((len(arr) - middle_length) / 2)
    for img_index in range(_, _ + middle_length, 2):
        batch_x.append(arr[img_index])
        batch_y.append(arr[img_index + 20])
    return batch_x, batch_y


def _crop(batch_x: list, batch_y: list, patch_size: tuple):
    """
    给定两个list `batch_x`, `batch_y`，对其中的每一对image array进行crop操作
    :param batch_x: list
    :param batch_y: list
    :param patch_size: 生成patch的大小，(patch_height, patch_width)
    :return:
    """
    for i in range(len(batch_x)):
        _shape = batch_x[i].shape  # [512, 512]
        _rand_x = random.randint(0, _shape[1] - patch_size[1] - 1)
        _rand_y = random.randint(0, _shape[0] - patch_size[0] - 1)
        _box = [_rand_x, _rand_y, _rand_x + patch_size[1], _rand_y + patch_size[0]]
        batch_x[i] = np.array(Image.fromarray(batch_x[i]).crop([*_box]))
        batch_y[i] = np.array(Image.fromarray(batch_y[i]).crop([*_box]))
    return batch_x, batch_y


def gen_batches(workspaces: list):
    batch_x, batch_y = [], []
    for workspace in workspaces:
        # load patient's ct voxels
        print("[INFO] loading {}...".format(workspace))
        ct_workspace = os.path.join(workspace, "4")
        ct_filenames = [os.path.join(ct_workspace, _) for _ in os.listdir(ct_workspace) if _.startswith("CT_")]
        ct_filenames.sort(key=lambda _: int(_.split("_")[-1]))
        _x, _y = _gen_batches([pydicom.read_file(_).pixel_array for _ in ct_filenames], middle_length=150)
        batch_x.extend(_x)
        batch_y.extend(_y)
    print("cropping image...")
    batch_x, batch_y = _crop(batch_x, batch_y, (256, 256))
    print("stacking image...")
    batch_x = np.stack(batch_x, axis=0)
    batch_y = np.stack(batch_y, axis=0)
    print("reshaping image...")
    batch_x = batch_x.reshape(*batch_x.shape, 1)
    batch_y = batch_y.reshape(*batch_y.shape, 1)
    # show batches
    # sitk.Show(sitk.GetImageFromArray(batch_x))
    # sitk.Show(sitk.GetImageFromArray(batch_y))
    # return batches
    return batch_x, batch_y


def main():
    workspaces = [os.path.join(r"F:\registration", _) for _ in os.listdir(r"F:\registration") if not _.endswith("ckle")]
    pickle.dump(gen_batches(workspaces), open(r"F:\registration\ct_batches.pickle", 'wb'))


if __name__ == '__main__':
    main()
