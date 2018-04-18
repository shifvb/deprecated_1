import os
import pickle
import tensorflow as tf
import SimpleITK as sitk
from DIRNet_tensorflow_master.models.models import DIRNet


def my_test():
    """暂时往里面验证一些图像"""
    # 加载数据
    batch_x, batch_y = pickle.load(open(r"F:\registration_patches\向水平或竖直方向移动8-13像素\test\ct_batches_test.pickle", 'rb'))
    # batch_x = batch_x / 255
    # batch_y = batch_y / 255
    batch_x = batch_x[0:len(batch_x): 30]
    batch_y = batch_y[0:len(batch_y): 30]
    batch_x = (batch_x - batch_x.min()) / (batch_x.max() - batch_x.min())
    batch_y = (batch_y - batch_y.min()) / (batch_y.max() - batch_y.min())
    # show image
    # sitk.Show(sitk.GetImageFromArray(batch_x))
    # sitk.Show(sitk.GetImageFromArray(batch_y))
    # exit()

    # config
    config_dict = {
        "checkpoint_dir": r"F:\registration_running_data\checkpoints",
        "image_size": [128, 128],
        "batch_size": len(batch_x),
        "result_dir": r"F:\registration_running_data\result",
    }

    # file operations
    if not os.path.exists(config_dict["result_dir"]):
        os.mkdir(config_dict["result_dir"])

    # create network
    sess = tf.Session()
    reg = DIRNet(sess, config_dict, "DIRNet", is_train=False)
    reg.restore(config_dict["checkpoint_dir"])

    # register
    reg.deploy(config_dict["result_dir"], batch_x, batch_y)


if __name__ == "__main__":
    my_test()
