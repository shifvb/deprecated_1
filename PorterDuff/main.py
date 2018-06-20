import os
import numpy as np
from PIL import Image
from PorterDuff.PorterDuff import PorterDuff
from PorterDuff.gray2color import gray2color


def main():
    mov_path = r"C:\Users\anonymous\Desktop\2\mov.png"
    fix_path = r"C:\Users\anonymous\Desktop\2\fix.png"
    mov_img = Image.open(mov_path).convert(mode="RGBA")
    fix_img = Image.open(fix_path).convert(mode="RGBA")

    mov_arr = np.array(mov_img)
    fix_arr = np.array(fix_img)


    mov_arr[:, :, :3] = mov_arr[:, :, :3] / (mov_arr[:, :, :3].max() - mov_arr[:, :, :3].min()) * 255
    # mov_arr[:, :, :3] =np.clip((mov_arr[:, :, :3].astype(np.int32)) + 120, 0, 255).astype(np.uint8)
    mov_arr = mov_arr.astype(np.uint8)
    mov_arr[:, :, :3] = gray2color(mov_arr[:, :, 2])  # R

    # fix_arr[:, :, :3] = fix_arr[:, :, :3] / (fix_arr[:, :, :3].max() - fix_arr[:, :, :3].min()) * 255
    # fix_arr = mov_arr.astype(np.uint8)
    # fix_arr[:, :, :3] = gray2color(fix_arr[:, :, 2])  # R

    mov_arr = mov_arr.astype(np.float32)
    fix_arr = fix_arr.astype(np.float32)

    # alpha
    mov_arr[:, :, 3] = 255 * 0.4
    fix_arr[:, :, 3] = 255 * 1
    # # # R
    # mov_arr[:, :, 1] = 0
    # mov_arr[:, :, 2] = 0
    # # # G
    # fix_arr[:, :, 0] = 0
    # fix_arr[:, :, 2] = 0
    print(mov_arr.shape, mov_arr.dtype)
    print(fix_arr.shape, fix_arr.dtype)

    pd = PorterDuff(mov_arr, fix_arr)
    out_arr = pd.alpha_composition(mode=PorterDuff.OVERLAY)
    Image.fromarray(mov_arr.astype(np.uint8)).save(r"C:\Users\anonymous\Desktop\2\mov_out.png")
    Image.fromarray(fix_arr.astype(np.uint8)).save(r"C:\Users\anonymous\Desktop\2\fix_out.png")
    Image.fromarray(out_arr).save(r"C:\Users\anonymous\Desktop\2\out.png")
    # Image.fromarray(mov_arr.astype(np.uint8)).save(r"C:\Users\anonymous\Desktop\2\mov_out.png")


if __name__ == '__main__':
    main()
