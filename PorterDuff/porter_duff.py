import os
import numpy as np
from PIL import Image


class PorterDuff(object):
    """
    PorterDuff python implementation
    --------------
    References:
        [1] https://en.wikipedia.org/wiki/Alpha_compositing
        [2] https://www.jianshu.com/p/d11892bbe055
    """
    CLEAR = 0  # [0, 0]
    SRC = 1  # [Sa, Sc]
    DST = 2  # [Da, Dc]
    SRC_OVER = 3  # [Sa + (1 - Sa)*Da, Rc = Sc + (1 - Sa)*Dc]
    DST_OVER = 4  # [Sa + (1 - Sa)*Da, Rc = Dc + (1 - Da)*Sc]
    SRC_IN = 5  # [Sa * Da, Sc * Da]
    DST_IN = 6  # [Sa * Da, Sa * Dc]
    SRC_OUT = 7  # [Sa * (1 - Da), Sc * (1 - Da)]
    DST_OUT = 8  # [Da * (1 - Sa), Dc * (1 - Sa)]
    SRC_ATOP = 9  # [Da, Sc * Da + (1 - Sa) * Dc]
    DST_ATOP = 10  # [Sa, Sa * Dc + Sc * (1 - Da)]
    XOR = 11  # [Sa + Da - 2 * Sa * Da, Sc * (1 - Da) + (1 - Sa) * Dc]
    DARKEN = 12  # [Sa + Da - Sa*Da, Sc*(1 - Da) + Dc*(1 - Sa) + min(Sc, Dc)]
    LIGHTEN = 13  # [Sa + Da - Sa*Da, Sc*(1 - Da) + Dc*(1 - Sa) + max(Sc, Dc)]
    MULTIPLY = 14  # [Sa * Da, Sc * Dc]
    SCREEN = 15  # [Sa + Da - Sa * Da, Sc + Dc - Sc * Dc]
    ADD = 16  # Saturate(S + D)
    OVERLAY = 17

    def __init__(self, source_arr, destination_arr):
        # More information of straight (unassociated) alpha, and premultiplied (associated) alpha can be seen at [1]
        self._Sc = (source_arr[:, :, :-1] / 255).astype(np.float32)
        self._Sa = (source_arr[:, :, -1:] / 255).astype(np.float32)
        self._Sc = self._Sc * self._Sa  # premultiplied (associated) alpha
        self._Dc = (destination_arr[:, :, :-1] / 255).astype(np.float32)
        self._Da = (destination_arr[:, :, -1:] / 255).astype(np.float32)
        self._Dc = self._Dc * self._Da  # premultiplied (associated) alpha

        self._out_color = None
        self._out_alpha = None

    def alpha_composition(self, mode):
        if mode == PorterDuff.CLEAR:
            self._clear_mode()
        elif mode == PorterDuff.SRC:
            self._src_mode()
        elif mode == PorterDuff.DST:
            self._dst_mode()
        elif mode == PorterDuff.SRC_OVER:
            self._src_over_mode()
        elif mode == PorterDuff.DST_OVER:
            self._dst_over_mode()
        else:
            raise ValueError("Not a Valid Mode: {}".format(mode))

        return np.concatenate(
            [(self._out_color * 255).astype(np.uint8), (self._out_alpha * 255).astype(np.uint8)],
            axis=2
        )

    def _clear_mode(self):  # CLEAR = 0  # [0, 0]
        self._out_alpha = np.ones_like(self._Sa, dtype=np.float32)
        self._out_color = np.ones_like(self._Sc, dtype=np.float32)

    def _src_mode(self):  # SRC = 1  # [Sa, Sc]
        self._out_alpha = self._Sa
        self._out_color = self._Sc

    def _dst_mode(self):  # DST = 2  # [Da, Dc]
        self._out_alpha = self._Da
        self._out_color = self._Dc

    def _src_over_mode(self):  # [Sa + (1 - Sa)*Da, Rc = Sc + (1 - Sa)*Dc]
        self._out_alpha = self._Sa + (1 - self._Sa) * self._Da
        self._out_color = self._Sc + (1 - self._Sa) * self._Dc

    def _dst_over_mode(self):  # [Sa + (1 - Sa)*Da, Rc = Dc + (1 - Da)*Sc]
        self._out_alpha = self._Sa + (1 - self._Sa) * self._Da
        self._out_color = self._Dc + (1 - self._Da) * self._Sc

    # def _mode(self):  #
    #     self._out_alpha =
    #     self._out_color =


def porter_duff(mode):
    _pd = PorterDuff(source_arr, destination_arr)
    return _pd.alpha_composition(mode)


if __name__ == '__main__':
    source_img = Image.open(r"C:\Users\anonymous\Desktop\1\source.png").convert(mode='RGBA')
    destination_img = Image.open(r"C:\Users\anonymous\Desktop\1\destination.png").convert(mode='RGBA')
    source_arr = np.array(source_img)
    destination_arr = np.array(destination_img)

    out_path = r'C:\Users\anonymous\Desktop\1\out.png'
    out_arr = porter_duff(PorterDuff.DST_OVER)
    Image.fromarray(out_arr, "RGBA").save(out_path)
