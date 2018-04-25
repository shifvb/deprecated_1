import tensorflow as tf
from 日文论文实现.models.conv_regressor import ConvNetRegressor


def deploy():
    config_dict = config_folder_guard({
        # network settings
        "batch_size": 3,
        "img_height": 512,
        "img_width": 512,

        # folder path
        "checkpoint_folder": "",  # todo remove it
        "test_visualization_folder": "",  # todo change it
    })
    sess = tf.Session()
    reg = ConvNetRegressor(sess, is_train=False, config=config_dict)
    reg.restore(sess, config_dict["checkpoint_folder"])
    batch_x, batch_y = get_deploy_batches(config_dict["batch_size"])
    reg.deploy(config_dict["test_visualization_folder"], batch_x, batch_y)


def get_deploy_batches(batch_size: int):
    batch_x, batch_y = None, None  # todo change it
    return batch_x, batch_y


def config_folder_guard(config_dict: dict):
    """防止出现文件夹不存在的情况"""
    pass  # todo: change it, do some guard things
    return config_dict


if __name__ == '__main__':
    deploy()
