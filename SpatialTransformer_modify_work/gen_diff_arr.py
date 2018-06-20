from PIL import Image
import numpy as np


def gen_diff_arr(x_path, y_path, out_path):
    x_arr = np.array(Image.open(x_path)).astype(np.int32)
    y_arr = np.array(Image.open(y_path)).astype(np.int32)
    diff_arr = np.abs(x_arr - y_arr).astype(np.uint8)
    Image.fromarray(diff_arr, "L").save(out_path)


if __name__ == '__main__':
    gen_diff_arr(
        x_path=r"F:\tmp3\transformed.png",
        y_path=r"F:\tmp3\original.png",
        out_path=r"f:\tmp3\diff.png"
    )
