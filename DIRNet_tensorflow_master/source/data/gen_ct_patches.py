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
    :return:
    """
    assert len(arr) > middle_length
    batch_x, batch_y = [], []
    _ = int((len(arr) - middle_length) / 2)
    for img_index in range(_, _ + middle_length, 2):
        batch_x.append(arr[img_index])
        batch_y.append(arr[img_index + 1])
    return batch_x, batch_y


def gen_batches(workspaces: list):
    batch_x, batch_y = [], []
    for workspace in workspaces:
        # load patient's ct voxels
        print("[INFO] loading {}...".format(workspace))
        ct_workspace = os.path.join(workspace, "4")
        ct_filenames = [os.path.join(ct_workspace, _) for _ in os.listdir(ct_workspace) if _.startswith("CT_")]
        ct_filenames.sort(key=lambda _: int(_.split("_")[-1]))
        _x, _y = _gen_batches([pydicom.read_file(_).pixel_array for _ in ct_filenames], middle_length=200)
        batch_x.extend(_x)
        batch_y.extend(_y)
    batch_x = np.stack(batch_x, axis=0)
    batch_y = np.stack(batch_y, axis=0)
    batch_x = batch_x.reshape(*batch_x.shape, 1)
    batch_y = batch_y.reshape(*batch_y.shape, 1)

    # generate batches
    return batch_x, batch_y


def main():
    workspaces = [os.path.join(r"F:\registration", _) for _ in os.listdir(r"F:\registration")]
    _ = gen_batches(workspaces)
    out_path = r"F:\registration\ct_batches.pickle"
    if not os.path.exists(out_path):
        pickle.dump(_, open(out_path, 'wb'))


if __name__ == '__main__':
    main()
