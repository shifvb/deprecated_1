import os
import shutil
import pydicom as dicom


def transport(input_folder_path: str, output_folder_path: str):
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
    # check folders
    if not os.path.exists(input_folder_path) or not os.path.isdir(input_folder_path):
        raise IOError("输入数据文件夹{} 不存在！".format(input_folder_path))
    if not os.path.exists(output_folder_path) or not os.path.isdir(output_folder_path):
        raise IOError("输出数据文件夹{} 不存在！".format(output_folder_path))
    if len(os.listdir(output_folder_path)) != 0:
        raise IOError("输出数据文件夹{} 非空！".format(output_folder_path))

    # get abs ct folder path
    if "4" in os.listdir(input_folder_path):
        ct_folder_path = os.path.join(input_folder_path, "4")
    elif "ct" in os.listdir(input_folder_path):
        ct_folder_path = os.path.join(input_folder_path, "ct")
    else:
        raise IOError("ct 文件夹未找到！")

    # get abs pt folder path
    if "5" in os.listdir(input_folder_path):
        pt_folder_path = os.path.join(input_folder_path, "5")
    elif "pet" in os.listdir(input_folder_path):
        pt_folder_path = os.path.join(input_folder_path, "pet")
    else:
        raise IOError("pt 文件夹未找到！")

    # get absolute filenames
    ct_filenames = [os.path.join(ct_folder_path, _) for _ in os.listdir(ct_folder_path) if not _.startswith("OT_")]
    pt_filenames = [os.path.join(pt_folder_path, _) for _ in os.listdir(pt_folder_path) if not _.startswith("OT_")]
    if not len(ct_filenames) == len(pt_filenames):
        raise IOError("ct图像数量({})应该和pt图像数量({})相等".format(len(ct_filenames), len(pt_filenames)))

    # transfer files
    patient_name = os.path.split(input_folder_path)[-1]
    os.mkdir(os.path.join(output_folder_path, patient_name))
    os.mkdir(os.path.join(output_folder_path, patient_name, "CT"))
    os.mkdir(os.path.join(output_folder_path, patient_name, "PT"))
    for abs_filename in ct_filenames + pt_filenames:
        ds = dicom.read_file(abs_filename)
        instance_number = ds.get('InstanceNumber')
        modality = ds.get('Modality')
        output_filename = "{}_{:>03}".format(modality, instance_number)
        abs_output_filename = os.path.join(output_folder_path, patient_name, modality, output_filename)
        shutil.copy(abs_filename, abs_output_filename)


def main():
    input_folder = r'F:\DICOM_Washing\workspace\PT00704-5'
    # input_folder = r'F:\DICOM_Washing\workspace\PT00998-2'
    output_folder = r'F:\DICOM_Washing\workspace_out'
    transport(input_folder, output_folder)


if __name__ == '__main__':
    main()
