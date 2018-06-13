import os


def config_folder_guard(config: dict):
    if not os.path.exists(config["ckpt_dir"]):
        os.makedirs(config["ckpt_dir"])
    if not os.path.exists(config["temp_dir"]):
        os.makedirs(config["temp_dir"])
    return config
