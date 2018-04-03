import os
import pickle
import numpy as np
import pydicom
import dicom_numpy
from look_labels_app.utils.register import register


def _load_Hu_SUV_data(ct_path: str, pt_path: str):
    """
    加载配准后suv值，hu值数据
    :param ct_path: ct文件夹绝对路径
    :param pt_path: pt文件夹绝对路径
    :return: Hu数组，SUV数组
    """
    # load Hu value & SUV value
    hu_arrs, registered_suv_arrs, suv_arrs = register(ct_path, pt_path)
    return hu_arrs, registered_suv_arrs, suv_arrs


def _load_CT_PET_data(ct_path: str, pt_path: str):
    """
    加载原始CT值和PET值
    :param ct_path: ct文件夹绝对路径
    :param pt_path: pt文件夹绝对路径
    :return: CT array, [img_num, 512, 512], int16
              PT array, [img_num, 128, 128], int16
    """
    ct_filenames = [os.path.join(ct_path, _) for _ in os.listdir(ct_path) if _.startswith("CT_")]
    ct_filenames.sort(key=lambda _: int(_.split("_")[-1]))
    pt_filenames = [os.path.join(pt_path, _) for _ in os.listdir(pt_path) if _.startswith("PT_")]
    pt_filenames.sort(key=lambda _: int(_.split("_")[-1]))
    # 拼接数组
    ct_voxel_ndarray, ct_ijk_to_xyz = dicom_numpy.combine_slices([pydicom.read_file(_) for _ in ct_filenames])
    pt_voxel_ndarray, pt_ijk_to_xyz = dicom_numpy.combine_slices([pydicom.read_file(_) for _ in pt_filenames])
    # 返回值
    return ct_voxel_ndarray.transpose([2, 1, 0])[::-1, :, :], pt_voxel_ndarray.transpose([2, 1, 0])[::-1, :, :], ct_ijk_to_xyz, pt_ijk_to_xyz


def load_data(ct_path: str, pt_path: str, work_directory: str):
    """加载数据(优先使用缓存), 缓存文件路径为 `{work_directory}/temp/registered.pickle`"""
    # 如果文件夹不存在创建文件夹
    _temp_dir = os.path.join(work_directory, "temp")
    if not os.path.isdir(_temp_dir):
        os.mkdir(_temp_dir)
    # 如果没有缓存，那么就写入缓存
    _temp_filename = os.path.join(_temp_dir, "registered.pickle")
    if not os.path.exists(_temp_filename):
        pickle.dump(_load_CT_PET_data(ct_path, pt_path), open(_temp_filename, 'wb'))
    # 读取缓存
    return pickle.load(open(_temp_filename, 'rb'))


if __name__ == '__main__':
    _load_CT_PET_data(r'F:\registration\PT38875\4', r'F:\registration\PT38875\5')
