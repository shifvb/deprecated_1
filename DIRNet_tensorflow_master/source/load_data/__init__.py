import os
import random
import numpy as np
import pydicom
from PIL import Image
import SimpleITK as sitk


def gen_patch(workspace: str, out_path: str, ):
    """

    :param workspace:
    :param out_path:
    :return:
    """
    # load filenames/series
    ct_workspace = os.path.join(workspace, "4")
    pt_workspace = os.path.join(workspace, "5")
    ct_series = [filename for filename in os.listdir(ct_workspace) if filename.startswith("CT_")]
    pt_series = [filename for filename in os.listdir(pt_workspace) if filename.startswith("PT_")]
    ct_series.sort(key=lambda filename: int(filename.split("_")[-1]))
    pt_series.sort(key=lambda filename: int(filename.split("_")[-1]))
    assert len(ct_series) == len(pt_series)

    # choose a proper file
    _index = random.choice(range(len(ct_series)))
    _index = 100
    ct_filename = ct_series[_index]
    pt_filename = pt_series[_index]
    assert ct_filename[1:] == pt_filename[1:]

    # load file dataset
    ct_ds = pydicom.read_file(os.path.join(ct_workspace, ct_filename))
    pt_ds = pydicom.read_file(os.path.join(pt_workspace, pt_filename))
    ct_img = Image.fromarray(norm_image(ct_ds.pixel_array)).resize([128, 128])
    pt_img = Image.fromarray(norm_image(pt_ds.pixel_array))
    # ct_img.save("ct_img_original.png")
    # pt_img.save("pt_img_original.png")
    ct_arr = np.array(ct_img)
    pt_arr = np.array(pt_img)

    # generate a region
    img_height, img_width = 128, 128
    patch_height, patch_width = 64, 64
    rect_left = random.randint(0, img_height - patch_height - 1), random.randint(0, img_width - patch_width - 1)
    patch_ct_arr = ct_arr[rect_left[0]: rect_left[0] + patch_height, rect_left[1]: rect_left[1] + patch_width]
    patch_pt_arr = pt_arr[rect_left[0]: rect_left[0] + patch_height, rect_left[1]: rect_left[1] + patch_width]
    Image.fromarray(patch_ct_arr).show()
    Image.fromarray(patch_pt_arr).show()

    ct_arr_show = np.stack([ct_arr for _ in range(3)], axis=2)
    pt_arr_show = np.stack([pt_arr for _ in range(3)], axis=2)
    for row in [rect_left[0], rect_left[0] + patch_height]:
        for col in range(rect_left[1], rect_left[1] + patch_width):
            ct_arr_show[row, col] = (255, 0, 0)
            pt_arr_show[row, col] = (255, 0, 0)
    for col in [rect_left[1], rect_left[1] + patch_width]:
        for row in range(rect_left[0], rect_left[0] + patch_height):
            ct_arr_show[row, col] = (255, 0, 0)
            pt_arr_show[row, col] = (255, 0, 0)
    Image.fromarray(ct_arr_show).show()
    Image.fromarray(pt_arr_show).show()


def norm_image(arr: np.ndarray):
    """
    将一个numpy数组正则化（0~255）,并转成np.uint8类型
    :param arr: 要处理的numpy数组
    :return: 值域在0~255之间的uint8数组
    """
    if not arr.min() == arr.max():
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return np.array(arr, dtype=np.uint8)


if __name__ == '__main__':
    gen_patch(workspace=r"F:\registration\PT38875",
              out_path=r"F:\registration_patches\PT38875", )
    # print(norm_image(np.array([1,2,3])))
