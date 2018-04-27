import os
import pickle
import tensorflow as tf
import SimpleITK as sitk
from DIRNet_tensorflow_master.models.models import DIRNet


def my_test():
    # 加载数据
    batch_x, batch_y = pickle.load(open(r"F:\registration_patches\向右移动11像素\test\ct_batches_test.pickle", 'rb'))

    # config
    config_dict = test_config_guard({
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "image_size": [128, 128],
        "batch_size": 10,
        "result_dir": r"F:\registration_running_data\result",
    })
    # create network
    sess = tf.Session()
    reg = DIRNet(sess, config_dict, "DIRNet", is_train=False)
    reg.restore(config_dict["checkpoint_dir"])

    # register
    _iter_num = len(batch_x) // config_dict["batch_size"]
    for i in range(_iter_num):
        _start_index, _end_index = i * config_dict["batch_size"], (i + 1) * config_dict["batch_size"]
        _batch_x = batch_x[_start_index:_end_index]
        _batch_y = batch_y[_start_index:_end_index]
        _batch_x = (_batch_x - _batch_x.min()) / (_batch_x.max() - _batch_x.min())
        _batch_y = (_batch_y - _batch_y.min()) / (_batch_y.max() - _batch_y.min())
        reg.deploy(config_dict["result_dir"], _batch_x, _batch_y, img_name_start_idx=int(i * config_dict["batch_size"]))


def test_config_guard(config_dict: dict):
    if not os.path.exists(config_dict["result_dir"]):
        os.makedirs(config_dict["result_dir"])
    if not os.path.exists(config_dict["checkpoint_dir"]):
        raise IOError("checkpoint_dir does not exists!")
    return config_dict


if __name__ == "__main__":
    my_test()
