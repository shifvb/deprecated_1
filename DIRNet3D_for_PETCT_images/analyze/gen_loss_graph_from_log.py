import os
import numpy as np
import matplotlib.pyplot as plt


def gen_loss_graph_from_log_v4():
    workspace = r"F:\registration_running_data\log"
    train_log = os.path.join(workspace, "train.log")
    valid_log = os.path.join(workspace, "valid.log")
    train_iter_interval = 400

    with open(train_log, 'r') as f:
        train_y_text = f.read()
    with open(valid_log, 'r') as f:
        valid_y_text = f.read()

    # 获取y座标
    train_y_text = [_.strip() for _ in train_y_text.split("\n") if _ != ""]
    valid_y_text = [_.strip() for _ in valid_y_text.split("\n") if _ != ""]
    train_loss_list = [float(_.split(",")[1].split("=")[-1]) for _ in train_y_text]
    valid_loss_list = [float(_.split(",")[1].split("=")[-1]) for _ in valid_y_text]
    train_ncc_list = [float(_.split(",")[2].split("=")[-1]) for _ in train_y_text]
    valid_ncc_list = [float(_.split(",")[2].split("=")[-1]) for _ in valid_y_text]
    train_grad_list = [float(_.split(",")[3].split("=")[-1]) * 0.001 for _ in train_y_text]
    valid_grad_list = [float(_.split(",")[3].split("=")[-1]) * 0.001 for _ in valid_y_text]

    # 获取x座标
    train_x_list = np.array(range(len(train_loss_list))) * train_iter_interval
    valid_x_list = np.array(range(len(valid_loss_list))) * train_iter_interval

    # 画图
    # plt.plot(train_x_list, train_loss_list, c="red", label='train_loss')
    # plt.plot(valid_x_list, valid_loss_list, c="blue", label="valid_loss")
    plt.plot(train_x_list, train_ncc_list, c="yellow", label='train_ncc')
    plt.plot(valid_x_list, valid_ncc_list, c="green", label="valid_ncc")
    # plt.plot(train_x_list, train_grad_list, c="cyan", label='train_grad')
    # plt.plot(valid_x_list, valid_grad_list, c="purple", label="valid_grad")
    # 配置
    plt.xlabel("iteration")
    plt.ylabel("loss")
    # plt.axis([0, 20000, -0.996, -0.984])
    plt.legend(bbox_to_anchor=[0.9, 0.9])
    plt.show()


if __name__ == '__main__':
    gen_loss_graph_from_log_v4()
