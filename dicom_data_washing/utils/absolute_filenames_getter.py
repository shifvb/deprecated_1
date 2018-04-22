import os


class AbsFileNamesGetter(object):
    def __init__(self, path: str):
        self._path = path

    def get(self):
        _ct_filenames = self._get_ct()
        _pt_filenames = self._get_pt()
        if not len(_ct_filenames) == len(_pt_filenames):
            raise IOError("ct图像数量({})应该和pt图像数量({})相等".format(len(_ct_filenames), len(_pt_filenames)))
        return _ct_filenames, _pt_filenames

    def _get_ct(self):  # get abs ct folder path
        if "4" in os.listdir(self._path):
            ct_folder_path = os.path.join(self._path, "4")
        elif "ct" in os.listdir(self._path):
            ct_folder_path = os.path.join(self._path, "ct")
        else:
            raise IOError("ct 文件夹未找到！")
        return [os.path.join(ct_folder_path, _) for _ in os.listdir(ct_folder_path) if not _.startswith("OT_")]

    def _get_pt(self):  # get abs pt folder path
        if "5" in os.listdir(self._path):
            pt_folder_path = os.path.join(self._path, "5")
        elif "pet" in os.listdir(self._path):
            pt_folder_path = os.path.join(self._path, "pet")
        else:
            raise IOError("pt 文件夹未找到！")
        return [os.path.join(pt_folder_path, _) for _ in os.listdir(pt_folder_path) if not _.startswith("OT_")]
