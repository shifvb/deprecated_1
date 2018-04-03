import os
import pickle
from look_labels_app.utils.register import register


def _load_data(ct_path: str, pt_path: str, mask_path: str):
    """
    加载配准后suv值，hu值和mask值数据
    :param ct_path: ct文件夹绝对路径
    :param pt_path: pt文件夹绝对路径
    :param mask_path: mask文件夹绝对路径
    :return: Hu数组，SUV数组，mask数组
    """
    # load Hu value & SUV value
    hu_arrs, registered_suv_arrs, suv_arrs = register(ct_path, pt_path)
    return hu_arrs, registered_suv_arrs, suv_arrs


def load_data(ct_path: str, pt_path: str, mask_path: str, work_directory: str):
    """加载数据(优先使用缓存), 缓存文件路径为 {work_directory/temp/registered.pickle}
    详细介绍见函数 _load_data()"""
    # 如果文件夹不存在创建文件夹
    _temp_dir = os.path.join(work_directory, "temp")
    if not os.path.isdir(_temp_dir):
        os.mkdir(_temp_dir)
    # 如果没有缓存，那么就写入缓存
    _temp_filename = os.path.join(_temp_dir, "registered.pickle")
    if not os.path.exists(_temp_filename):
        pickle.dump(_load_data(ct_path, pt_path, mask_path), open(_temp_filename, 'wb'))
    # 读取缓存
    return pickle.load(open(_temp_filename, 'rb'))
