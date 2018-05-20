from mnist import MNIST
import numpy as np
from PIL import Image
import os


def gen_train():
    workspace = r'F:\temp2_mnist\train'
    for i in range(10):
        if not os.path.exists(os.path.join(workspace, str(i))):
            os.makedirs(os.path.join(workspace, str(i)))

    mndata = MNIST('.')
    images, labels = mndata.load_training()
    for i in range(len(images)):
        img_arr = np.array(images[i]).reshape([28, 28]).astype(np.uint8)
        _path = os.path.join(workspace, str(labels[i]), "{}_{:>05}.png".format(labels[i], i))
        Image.fromarray(img_arr).save(_path)


def gen_test():
    workspace = r'F:\temp2_mnist\test'
    if not os.path.exists(workspace):
        os.makedirs(workspace)
    for i in range(10):
        if not os.path.exists(os.path.join(workspace, str(i))):
            os.makedirs(os.path.join(workspace, str(i)))

    mndata = MNIST('.')
    images, labels = mndata.load_testing()
    for i in range(len(images)):
        img_arr = np.array(images[i]).reshape([28, 28]).astype(np.uint8)
        _path = os.path.join(workspace, str(labels[i]), "{}_{:>05}.png".format(labels[i], i))
        Image.fromarray(img_arr).save(_path)

if __name__ == '__main__':
    # gen_train()
    gen_test()