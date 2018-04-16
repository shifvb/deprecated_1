import matplotlib.pyplot as plt

# 通过粘贴日志产生图表
log_len = 10000
s = """"""


def main():
    assert log_len == 10000  # 默认训练次数
    assert log_len == len(s)
    global s
    s = [_ for _ in s.split('\n') if not _ == ""]
    s = [float(_.split(":")[-1]) for _ in s]
    plt.plot(range(log_len), s)
    plt.show()


if __name__ == '__main__':
    main()
