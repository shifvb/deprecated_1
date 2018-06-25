import os
import numpy as np
from PIL import Image
from PorterDuff_algo.PorterDuff import PorterDuff
from PorterDuff_algo.gray2color import gray2color


def main(mov_arr, fix_arr):
    mov_arr_c = mov_arr[:, :, :3]
    mov_arr_a = mov_arr[:, :, 3:]

    # 去除暗的
    # low_gray_level_pos = mov_arrmov_arr[:, :, :3]

    # 归一化
    mov_arr_c = mov_arr_c / (mov_arr_c.max() - mov_arr_c.min()) * 255

    # 生成伪彩色
    mov_arr_c = mov_arr_c.astype(np.uint8)
    mov_arr_c = gray2color(mov_arr_c[:, :, 0])  # R

    # 调整alpha值
    mov_arr_a = np.ones_like(mov_arr_a, dtype=np.float32) * 255 * 0.4
    mov_arr = np.concatenate([mov_arr_c, mov_arr_a], axis=2)
    print(mov_arr.dtype)

    # 图像融合
    pd = PorterDuff(mov_arr, fix_arr)
    out_arr = pd.alpha_composition(mode=PorterDuff.DARKEN)
    Image.fromarray(mov_arr.astype(np.uint8)).save(r"C:\Users\anonymous\Desktop\2\mov_out.png")
    Image.fromarray(fix_arr.astype(np.uint8)).save(r"C:\Users\anonymous\Desktop\2\fix_out.png")
    Image.fromarray(out_arr).save(r"C:\Users\anonymous\Desktop\2\out.png")
    # Image.fromarray(mov_arr.astype(np.uint8)).save(r"C:\Users\anonymous\Desktop\2\mov_out.png")


if __name__ == '__main__':
    mov_path = r"C:\Users\anonymous\Desktop\2\mov.png"
    fix_path = r"C:\Users\anonymous\Desktop\2\fix.png"
    mov_img = Image.open(mov_path).convert(mode="RGBA")
    fix_img = Image.open(fix_path).convert(mode="RGBA")
    main(np.array(mov_img), np.array(fix_img))
