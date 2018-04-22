import os


class PatientIDFolderGuard(object):
    def __init__(self, input_folder_path: str, output_folder_path: str,
                 output_ct_folder_name="CT", output_pt_folder_name="PT"):
        self._i = input_folder_path
        self._o = output_folder_path
        self._ct_folder_name = output_ct_folder_name
        self._pt_folder_name = output_pt_folder_name

    def guard(self):
        # check input folders
        if not os.path.exists(self._i) or not os.path.isdir(self._i):
            raise IOError("输入数据文件夹{} 不存在！".format(self._i))
        # check output folders
        if not os.path.exists(self._o) or not os.path.isdir(self._o):
            raise IOError("输出数据文件夹{} 不存在！".format(self._o))
        # check patient folders
        out_patient_folder_path = os.path.join(self._o, os.path.split(self._i)[-1])
        if os.path.exists(out_patient_folder_path):
            print("[INFO] 输出病例文件夹 \"{}\"存在，跳过".format(out_patient_folder_path))
            return False
        else:
            os.mkdir(out_patient_folder_path)
            os.mkdir(os.path.join(out_patient_folder_path, self._ct_folder_name))
            os.mkdir(os.path.join(out_patient_folder_path, self._pt_folder_name))
            print('[INFO] processing "{}" -> "{}"...'.format(self._i, out_patient_folder_path))
            return True
