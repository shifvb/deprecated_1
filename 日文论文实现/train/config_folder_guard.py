import os


def config_folder_guard(config_dict: dict):
    """防止出现文件夹不存在的情况"""
    # 检查输出文件夹是否存在，如果不存在则创建相应文件夹
    if not os.path.exists(config_dict["checkpoint_dir"]):
        os.makedirs(config_dict["checkpoint_dir"])
    if not os.path.exists(config_dict["valid_out_dir_1"]):
        os.makedirs(config_dict["valid_out_dir_1"])
    if not os.path.exists(config_dict["valid_out_dir_2"]):
        os.makedirs(config_dict["valid_out_dir_2"])
    if not os.path.exists(config_dict["valid_out_dir_3"]):
        os.makedirs(config_dict["valid_out_dir_3"])
    # if not os.path.exists(config_dict["valid_out_dir_all"]):
    #     os.makedirs(config_dict["valid_out_dir_all"])
    # 检查输入文件夹是否符合逻辑
    assert config_dict["valid_in_x_dir_1"] == config_dict["train_in_x_dir_1"]
    assert config_dict["valid_in_y_dir_1"] == config_dict["train_in_y_dir_1"]
    assert config_dict["train_in_x_dir_2"] == config_dict["valid_out_dir_1"]
    assert config_dict["valid_in_x_dir_2"] == config_dict["train_in_x_dir_2"]
    assert config_dict["valid_in_y_dir_2"] == config_dict["train_in_y_dir_2"]
    assert config_dict["train_in_x_dir_3"] == config_dict["valid_out_dir_2"]
    # assert config_dict["valid_in_x_dir_3"] == config_dict["train_in_x_dir_3"] # 这行代码确保R3验证集和R3训练集相同
    # assert config_dict["valid_in_x_dir_3"] == config_dict["train_in_x_dir_1"]  # 这行代码确保R3验证集和R1训练集(原图像)相同
    # assert config_dict["valid_in_y_dir_3"] == config_dict["train_in_y_dir_3"]
    # assert config_dict["train_in_x_dir_all"] == config_dict["valid_out_dir_3"]
    # assert config_dict["valid_in_x_dir_all"] == config_dict["train_in_x_dir_all"]
    # assert config_dict["valid_in_y_dir_all"] == config_dict["train_in_y_dir_all"]
    assert config_dict["train_in_y_dir_1"] == \
           config_dict["valid_in_y_dir_1"] == \
           config_dict["train_in_y_dir_2"] == \
           config_dict["valid_in_y_dir_2"] == \
           config_dict["train_in_y_dir_3"]
           # config_dict["valid_in_y_dir_3"]
           # config_dict["train_in_y_dir_all"] == \
           # config_dict["valid_in_y_dir_all"]
    return config_dict
