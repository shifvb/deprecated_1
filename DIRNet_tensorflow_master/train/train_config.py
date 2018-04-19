import os


def train_config_v1():
    """get config for training version 1"""
    _config = {
        # train batch folder
        "batch_folder": r"F:\registration_patches\向水平或竖直方向移动8-13像素\train",
        # train parameters
        "image_size": [128, 128],
        "batch_size": 80,
        "learning_rate": 1e-5,
        "iteration_num": 20000,
        # train data folder
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "temp_dir": r"F:\registration_running_data\temp",
        "logger_dir": r"f:\registration_running_data\log",
        "logger_name": "train.log",
    }
    if not os.path.exists(_config["checkpoint_dir"]):
        os.makedirs(_config["checkpoint_dir"])
    if not os.path.exists(_config["temp_dir"]):
        os.makedirs(_config["temp_dir"])

    return _config
