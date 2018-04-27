import os
import matplotlib.pyplot as plt


def main():
    workspace = r"F:\registration_results_backup\2018_04_17_11_37_实验三_使用了源程序自带的mse作为loss, 图像水平垂直移动8~13像素)_imgnum=60410_imgsize=128x128_batch=80_iter=10000\log"
    with open(os.path.join(workspace, "train.log"), 'r') as f:
        s = f.read()

    s = [_ for _ in s.split('\n') if not _ == ""]
    s = [float(_.split(":")[-1]) for _ in s]
    assert 10000 == len(s), len(s)  # 默认训练次数
    plt.plot(range(len(s)), s)
    plt.show()


def gen_graph_from_loss_ver2():
    workspace = r"F:\registration_results_backup\2018_04_18_12_58_实验一_(使用了ncc+loss_term_2, 图像水平或垂直移动8~13像素)_imgnum=60410_imgsize=128x128_batch=10_iter=10000\log"
    with open(os.path.join(workspace, "train.log"), 'r') as f:
        s = f.read()
    s = [_ for _ in s.split('\n') if not _ == ""]
    s1 = [float(_.split(",")[1].split("=")[1]) for _ in s]
    s2 = [float(_.split(",")[2].split("=")[1]) for _ in s]
    s3 = [float(_.split(",")[3].split("=")[1]) for _ in s]
    print(s1)
    print(s2)
    print(s3)
    plt.plot(range(len(s1)), s1, c='red')
    plt.plot(range(len(s2)), s2, c='blue')
    plt.plot(range(len(s3)), s3, c='green')
    plt.show()


if __name__ == '__main__':
    gen_graph_from_loss_ver2()
