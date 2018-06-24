import numpy as np
from PIL import Image


class PorterDuff(object):
    """
    PorterDuff python implementation
    --------------
    References:
        [1] http://graphics.pixar.com/library/Compositing/paper.pdf
        [2] https://en.wikipedia.org/wiki/Alpha_compositing
        [3] https://www.jianshu.com/p/d11892bbe055
        [4] https://blog.csdn.net/IO_Field/article/details/78222527
        [5] https://blog.csdn.net/android_cmos/article/details/78907166
        [6] https://blog.csdn.net/u013085697/article/details/52096703
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
    OVERLAY = 17  # A_out = A_dst + A_scr - A_dst * A_src, C_out = 2 * C_dst * C_src (or other methods) (archive [4])

    def __init__(self, source_arr, destination_arr):
        """
        # More information of straight (unassociated) alpha, and premultiplied (associated) alpha can be seen at [2]
        :param source_arr: source image numpy array of shape [height, width, channels]
        :param destination_arr:  destination numpy array of shape [height, width, channels]
        """
        # range from [0, 255](uint8) to [0, 1](float32)
        self._Sa = (source_arr[:, :, -1:] / 255).astype(np.float32)  # source alpha
        self._Sc = (source_arr[:, :, :-1] / 255).astype(np.float32)  # source color
        self._Da = (destination_arr[:, :, -1:] / 255).astype(np.float32)  # destination alpha
        self._Dc = (destination_arr[:, :, :-1] / 255).astype(np.float32)  # destination color
        # straight(unassociated) alpha to premultiplied(associated) alpha
        self._Sc = self._Sc * self._Sa  # premultiplied (associated) alpha
        self._Dc = self._Dc * self._Da  # premultiplied (associated) alpha
        # declare output variables
        self._Oa = None  # output alpha
        self._Oc = None  # output color

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
        elif mode == PorterDuff.SRC_IN:
            self._src_in_mode()
        elif mode == PorterDuff.DST_IN:
            self._dst_in_mode()
        elif mode == PorterDuff.SRC_OUT:
            self._src_out_mode()
        elif mode == PorterDuff.DST_OUT:
            self._dst_out_mode()
        elif mode == PorterDuff.SRC_ATOP:
            self._src_atop_mode()
        elif mode == PorterDuff.DST_ATOP:
            self._dst_atop_mode()
        elif mode == PorterDuff.XOR:
            self._xor_mode()
        elif mode == PorterDuff.DARKEN:
            self._darken_mode()
        elif mode == PorterDuff.LIGHTEN:
            self._lighten_mode()
        elif mode == PorterDuff.MULTIPLY:
            self._multiply_mode()
        elif mode == PorterDuff.SCREEN:
            self._screen_mode()
        elif mode == PorterDuff.ADD:
            self._add_mode()
        elif mode == PorterDuff.OVERLAY:
            self._overlay_mode()
        else:
            raise ValueError("Not a Valid Mode: {}".format(mode))

        # premultiplied(associated) alpha to straight(unassociated) alpha
        # Because in numpy, (np.array(1)/np.array(0.0)).astype(np.uint8) = 0
        # So, any element in self._Oa is zero doesn't matters. Just don't care about it.
        self._Oc = self._Oc / self._Oa
        # range from [0, 1](float32) to [0, 255](uint8)
        self._Oc = (self._Oc * 255).astype(np.uint8)
        self._Oa = (self._Oa * 255).astype(np.uint8)

        return np.concatenate([self._Oc, self._Oa], axis=2)

    def _clear_mode(self):  # CLEAR = 0  # [0, 0]
        self._Oa = np.ones_like(self._Sa, dtype=np.float32)
        self._Oc = np.ones_like(self._Sc, dtype=np.float32)

    def _src_mode(self):  # SRC = 1  # [Sa, Sc]
        self._Oa = self._Sa
        self._Oc = self._Sc

    def _dst_mode(self):  # DST = 2  # [Da, Dc]
        self._Oa = self._Da
        self._Oc = self._Dc

    def _src_over_mode(self):  # [Sa + (1 - Sa)*Da, Rc = Sc + (1 - Sa)*Dc]
        self._Oa = self._Sa + (1 - self._Sa) * self._Da
        self._Oc = self._Sc + (1 - self._Sa) * self._Dc

    def _dst_over_mode(self):  # [Sa + (1 - Sa)*Da, Rc = Dc + (1 - Da)*Sc]
        self._Oa = self._Sa + (1 - self._Sa) * self._Da
        self._Oc = self._Dc + (1 - self._Da) * self._Sc

    def _src_in_mode(self):  # [Sa * Da, Sc * Da]
        self._Oa = self._Sa * self._Da
        self._Oc = self._Sc * self._Da

    def _dst_in_mode(self):  # [Sa * Da, Sa * Dc]
        self._Oa = self._Sa * self._Da
        self._Oc = self._Sa * self._Dc

    def _src_out_mode(self):  # [Sa * (1 - Da), Sc * (1 - Da)]
        self._Oa = self._Sa * (1 - self._Da)
        self._Oc = self._Sc * (1 - self._Da)

    def _dst_out_mode(self):  # [Da * (1 - Sa), Dc * (1 - Sa)]
        self._Oa = self._Da * (1 - self._Sa)
        self._Oc = self._Dc * (1 - self._Sa)

    def _src_atop_mode(self):  # [Da, Sc * Da + (1 - Sa) * Dc]
        self._Oa = self._Da
        self._Oc = self._Sc * self._Da + (1 - self._Sa) * self._Dc

    def _dst_atop_mode(self):  # [Sa, Sa * Dc + Sc * (1 - Da)]
        self._Oa = self._Sa
        self._Oc = self._Sa * self._Dc + self._Sc * (1 - self._Da)

    def _xor_mode(self):  # [Sa + Da - 2 * Sa * Da, Sc * (1 - Da) + (1 - Sa) * Dc]
        self._Oa = self._Sa + self._Da - 2 * self._Sa * self._Da
        self._Oc = self._Sc * (1 - self._Da) + (1 - self._Sa) * self._Dc

    def _darken_mode(self):  # [Sa + Da - Sa*Da, Sc*(1 - Da) + Dc*(1 - Sa) + min(Sc, Dc)]
        self._Oa = self._Sa + self._Da - self._Sa * self._Da
        self._Oc = self._Sc * (1 - self._Da) + self._Dc * (1 - self._Sa) + np.min([self._Sc, self._Dc], axis=0)

    def _lighten_mode(self):  # [Sa + Da - Sa*Da, Sc*(1 - Da) + Dc*(1 - Sa) + max(Sc, Dc)]
        self._Oa = self._Sa + self._Da - self._Sa * self._Da
        self._Oc = self._Sc * (1 - self._Da) + self._Dc * (1 - self._Sa) + np.max([self._Sc, self._Dc], axis=0)

    def _multiply_mode(self):  # [Sa * Da, Sc * Dc]
        self._Oa = self._Sa * self._Da
        self._Oc = self._Sc * self._Dc

    def _screen_mode(self):  # [Sa + Da - Sa * Da, Sc + Dc - Sc * Dc]
        self._Oa = self._Sa + self._Da - self._Sa * self._Da
        self._Oc = self._Sc + self._Dc - self._Sc * self._Dc

    def _add_mode(self):  # Saturate(S + D)
        raise NotImplementedError()

    def _overlay_mode(self):
        # A_out = A_dst + A_scr - A_dst * A_src, C_out = 2 * C_dst * C_src (or other methods) (archive [4])
        self._Oa = self._Da + self._Sa - self._Da * self._Sa
        self._Oc = 2 * self._Dc * self._Sc


def porter_duff(x, y, mode):
    return PorterDuff(x, y).alpha_composition(mode)


if __name__ == '__main__':
    source_img = Image.open(r"img\source.png").convert(mode='RGBA')
    destination_img = Image.open(r"img\destination.png").convert(mode='RGBA')
    source_arr = np.array(source_img)
    destination_arr = np.array(destination_img)

    out_path = r'img\out.png'
    out_arr = porter_duff(source_arr, destination_arr, PorterDuff.DARKEN)
    Image.fromarray(out_arr, "RGBA").save(out_path)
