import os
import matplotlib.pyplot as plt


def main():
    workspace = r"F:\registration_results_backup\2018_04_16_17_00_实验二_向下移动11像素(获取数组reg.v)_imgnum=60410_imgsize=128x128_batch=80_iter=10000\log"
    with open(os.path.join(workspace, "train.log"), 'r') as f:
        s = f.read()

    s = [_ for _ in s.split('\n') if not _ == ""]
    s = [float(_.split(":")[-1]) for _ in s]
    assert 10000 == len(s), len(s)  # 默认训练次数
    plt.plot(range(len(s)), s)
    plt.show()


if __name__ == '__main__':
    main()
