import os
import pydicom
import math
import random
import pickle
import numpy as np
import SimpleITK as sitk
from PIL import Image


def _crop(batch_x: list, batch_y: list, patch_size: tuple, patch_num_per_img: int):
    """
    给定两个list `batch_x`, `batch_y`，对其中的每一对image array进行crop操作,做`patch_num_per_img`遍
    :param batch_x: list
    :param batch_y: list
    :param patch_size: 生成patch的大小，(patch_height, patch_width)
    :return:
    """
    _a, _b = [], []  # 要返回的 batch_x 和 batch_y。怕重名，换个名字
    _shape = batch_x[0].shape  # [512, 512]
    for i in range(len(batch_x)):
        for j in range(patch_num_per_img):
            # 随机产生x，y坐标
            _rand_x = random.randint(0, _shape[1] - patch_size[1] - 1)
            _rand_y = random.randint(0, _shape[0] - patch_size[0] - 1)
            # 产生moving image的偏移像素算法
            _rand_x_bias = random.choice([-9])
            _rand_y_bias = random.choice([0])
            # 分别计算 fixed_image 和 moving_image 的 box
            _fix_box = [_rand_x, _rand_y, _rand_x + patch_size[1], _rand_y + patch_size[0]]
            _mov_box = [_fix_box[0] + _rand_x_bias, _fix_box[1] + _rand_y_bias,
                        _fix_box[2] + _rand_x_bias, _fix_box[3] + _rand_y_bias]
            # 根据box截取图像(moving_image -> _patch_x, fixed_image -> _patch_y)
            _patch_x = np.array(Image.fromarray(batch_x[i]).crop([*_mov_box]), dtype=np.int16)
            _patch_y = np.array(Image.fromarray(batch_y[i]).crop([*_fix_box]), dtype=np.int16)
            # 存入数组
            _a.append(_patch_x)
            _b.append(_patch_y)
    # stack
    _a = np.stack(_a, axis=0)
    _b = np.stack(_b, axis=0)
    return _a, _b


def gen_batches(workspace: str, out_dir: str, out_name: str, patch_num_per_img: int, debug=False):
    # load patient's ct voxels
    print("[INFO] loading {}...".format(workspace))
    ct_workspace = os.path.join(workspace, "4")
    ct_filenames = [os.path.join(ct_workspace, _) for _ in os.listdir(ct_workspace) if _.startswith("CT_")]
    ct_filenames.sort(key=lambda _: int(_.split("_")[-1]))
    ct_arrs = [pydicom.read_file(_).pixel_array for _ in ct_filenames]
    print("[INFO] cropping image...")
    batch_x, batch_y = _crop(ct_arrs.copy(), ct_arrs.copy(), (128, 128), patch_num_per_img)
    if debug:
        print('[DEBUG] show batches...')
        sitk.Show(sitk.GetImageFromArray(batch_x))
        sitk.Show(sitk.GetImageFromArray(batch_y))
    print("[INFO] reshaping image...")
    batch_x = batch_x.reshape(*batch_x.shape, 1)
    batch_y = batch_y.reshape(*batch_y.shape, 1)
    print("[INFO] saving image...")
    with open(os.path.join(out_dir, out_name), 'wb') as f:
        pickle.dump((batch_x, batch_y), f)


def main():
    workspaces = [os.path.join(r"F:\registration", _) for _ in os.listdir(r"F:\registration")]

    # generate train patches
    train_workspaces = workspaces[:-1]
    train_out_dir = r"F:\registration_patches"
    train_out_name = r"ct_batches_train_{}.pickle"
    for i, train_workspace in enumerate(train_workspaces):
        gen_batches(train_workspace, train_out_dir, train_out_name.format(i), 10)

    # generate test patches
    test_workspace = workspaces[-1]
    test_out_dir = r"F:\registration_patches"
    test_out_name = r"ct_batches_test.pickle"
    gen_batches(test_workspace, test_out_dir, test_out_name, 10)


if __name__ == '__main__':
    main()
