def get_config(is_train: bool) -> dict:
    config_dict = {
        "checkpoint_dir": "checkpoint",
        "image_size": [28, 28],
    }
    # train configure
    if is_train:
        config_dict.update({
            "batch_size": 64,
            "learning_rate": 1e-4,
            "iteration_num": 1000,
            "temp_dir": "temp",
        })
    # deploy configure
    else:
        config_dict.update({
            "batch_size": 10,
            "result_dir": "result",
        })
    return config_dict
