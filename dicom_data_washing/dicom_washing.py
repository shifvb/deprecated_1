import os
import pydicom
from pydicom.errors import InvalidDicomError
from dicom_data_washing.utils import AbsFileNamesGetter, PatientIDFolderGuard, single_dicom_wash


def dicom_wash(input_folder_path: str, output_folder_path: str):
    """
    :param input_folder_path: 单个病例数据文件夹绝对路径，例如"C:\\data\\PT00000"
        要求输入文件夹具有如下结构：
        PT00000/
            ├── ct    (此处存放ct图像)
                ├── 341 (对于文件的要求，为升序数字即可)
                ├...
                ├── 681
            ├── pet     (此处存放pet图像)
                ├── 0 (对于文件的要求，为升序数字即可)
                 ...
                ├── 340
        或者有如下结构：
        PT00000/
            ├── 4
                ├── CT_001 (对于文件的要求，需要以`CT_`开头，后面接着序列号，如`001`)
                ├...
                ├── CT_249
            ├── 5
                ├── PT_001 (对于文件的要求，需要以`PT_`开头，后面接着序列号，如`001`)
                ├...
                ├── PT_249


    :param output_folder_path: 存放清洗过的数据文件夹名，例如"C:\\output_data\\"
    :return None
        输出文件夹具有以下结构：
        ├── output_data
            ├── PT00000
                ├── CT
                    ├── CT_001
                    ├...
                    ├── CT_249
                ├── PT
                    ├── PT_001
                    ├...
                    ├── PT_249
    """
    # folder guard
    if PatientIDFolderGuard(input_folder_path, output_folder_path).guard() is False:
        return  # 输出病例文件已存在就跳过
    # get absolute ct / pt filename
    abs_ct_filenames, abs_pt_filenames = AbsFileNamesGetter(input_folder_path).get()
    # wash data
    patient_name = os.path.split(input_folder_path)[-1]
    for abs_filename in abs_ct_filenames + abs_pt_filenames:
        ds, modality, output_filename = single_dicom_wash(pydicom.read_file(abs_filename))
        abs_output_filename = os.path.join(output_folder_path, patient_name, modality, output_filename)
        pydicom.write_file(abs_output_filename, ds)


def main():
    workspace = r'F:\DICOM_Washing\backup'
    input_folders = [os.path.join(workspace, _) for _ in os.listdir(workspace)]
    output_folder = r'F:\DICOM_Washing\workspace_out'
    for input_folder in input_folders:
        dicom_wash(input_folder, output_folder)


if __name__ == '__main__':
    main()
