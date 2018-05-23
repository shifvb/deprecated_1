import os
import matplotlib.pyplot as plt


def gen_loss_graph_from_log_version_3():
    workspace = r"F:\registration_running_data\log"
    train_log = os.path.join(workspace, "train.log")
    valid_log = os.path.join(workspace, "valid.log")

    with open(train_log, 'r') as f:
        train_text = f.read()
    with open(valid_log, 'r') as f:
        valid_text = f.read()

    train_text = [_.strip() for _ in train_text.split("\n") if _ != ""]
    valid_text = [_.strip() for _ in valid_text.split("\n") if _ != ""]
    train_list = [float(_.split(",")[-1].split("=")[-1]) for _ in train_text]
    valid_list = [float(_.split(",")[-1].split("=")[-1]) for _ in valid_text]
    plt.plot(range(len(train_list)), train_list, c="red", label='train_loss(19450 img average)')
    plt.plot(range(len(valid_list)), valid_list, c="blue", label="valid_loss(6480 img average)")
    plt.xlabel("epoch")
    plt.ylabel("loss(-NCC)")
    plt.legend(bbox_to_anchor=[1, 1])
    plt.show()


if __name__ == '__main__':
    gen_loss_graph_from_log_version_3()
