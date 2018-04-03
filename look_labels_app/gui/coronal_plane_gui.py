import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from look_labels_app.utils.Clock import Clock
from look_labels_app.utils.ImageProcessor import norm_image, gen_fuse_arr


class CoronalPlaneGUI(tk.Toplevel):
    def __init__(self, hu_arrs: np.ndarray, suv_arrs: np.ndarray, ct_patient_info: dict, pt_patient_info: dict):
        super().__init__()
        # 窗口设置
        self._window_size = (500, 1024)  # height, width
        self.top_level = self
        self.top_level.title("CoronalPlaneView")
        self.top_level.geometry("1536x1000+0+0")
        self.top_level.bind("<Key-Left>", self.prev_page_callback)
        self.top_level.bind("<Key-Right>", self.next_page_callback)
        self.top_level.protocol("WM_DELETE_WINDOW", self.close_window_callback)
        self.top_level.focus_set()
        self.clock = Clock(0.1)

        # 压缩ct图像到128x128
        L = []
        for img_arr in hu_arrs:
            shrinked_img = Image.fromarray(img_arr).resize([128, 128], resample=Image.BICUBIC)
            L.append(np.array(shrinked_img, dtype=np.int16))
        hu_arrs = np.stack(L, axis=0)
        suv_arrs = suv_arrs.astype(np.float32)

        # 数据设置
        self.current_index = 0
        self.total_img_num = hu_arrs.shape[1]
        self.ct_patient_info = ct_patient_info
        self.pt_patient_info = pt_patient_info
        self.hu_arrs = self.from_transverse_plane_to_coronal_plane(hu_arrs, is_PT=True)
        self.suv_arrs = self.from_transverse_plane_to_coronal_plane(suv_arrs, is_PT=True)
        self.hu_arrs = norm_image(self.hu_arrs)
        self.suv_arrs = norm_image(self.suv_arrs)

        # hu frame
        hu_frame = tk.Frame(self.top_level)
        hu_frame.grid(row=0, column=0)
        self.ct_canvas = tk.Canvas(hu_frame, width=512, height=1000)
        self.ct_canvas.pack()
        # suv frame
        suv_frame = tk.Frame(self.top_level)
        suv_frame.grid(row=0, column=1)
        self.suv_canvas = tk.Canvas(suv_frame, width=512, height=1000)
        self.suv_canvas.pack()
        # right most frame
        right_frame = tk.Frame(self.top_level)
        right_frame.grid(row=0, column=1)

        # call method to load image
        self.load_images()

    @staticmethod
    def _threshold_image(arr, min_value=None, max_value=None):
        if min_value is not None:
            _ = arr < min_value
            arr *= (1 - _)
            arr += (_ * min_value)
        if max_value is not None:
            _ = arr > max_value
            arr *= (1 - _)
            arr += (_ * max_value)
        return arr

    def load_images(self):
        """在界面上加载图像"""
        # load arrays
        ct_arr = self.hu_arrs[self.current_index]
        suv_arr = self.suv_arrs[self.current_index]

        # load ct image
        self.current_ct_img = ImageTk.PhotoImage(self.resize_to_fit_screen(ct_arr))
        self.ct_canvas.create_image(0, 0, image=self.current_ct_img, anchor=tk.NW)
        # load suv image
        self.current_suv_img = ImageTk.PhotoImage(self.resize_to_fit_screen(suv_arr))
        self.suv_canvas.create_image(0, 0, image=self.current_suv_img, anchor=tk.NW)

        # set title
        self.top_level.title("CoronalPlaneView ({} / {})".format(self.current_index + 1, self.total_img_num))

    def from_transverse_plane_to_coronal_plane(self, arrs, is_PT=False):
        """从横断面转到冠状面进行数组转换，
        第一步是转秩变为冠状面，第二步是根据DICOM文件中实际像素间距，将图像拉伸"""
        # 转秩
        arrs = arrs.transpose([1, 0, 2])
        if not is_PT:
            # 图像拉伸
            _ratio = self.ct_patient_info['sliceThickness'] / self.ct_patient_info['pixelSpacing'][1]
            _old_size = arrs[0].shape
            _new_size = [int(_) for _ in (_old_size[0] * _ratio, _old_size[1])]
            _new_size.reverse()  # PIL.Image.resize() receive format of (width, height), rather than (height, width)
            return np.stack([np.array(Image.fromarray(_).resize(_new_size, Image.BICUBIC)) for _ in arrs], axis=0)
        else:
            # 图像拉伸
            _ratio = self.pt_patient_info['sliceThickness'] / self.pt_patient_info['pixelSpacing'][1]
            _old_size = arrs[0].shape
            _new_size = [int(_) for _ in (_old_size[0] * _ratio, _old_size[1])]
            _new_size.reverse()  # PIL.Image.resize() receive format of (width, height), rather than (height, width)
            return np.stack([np.array(Image.fromarray(_).resize(_new_size, Image.BICUBIC)) for _ in arrs], axis=0)

    def resize_to_fit_screen(self, arr: np.ndarray):
        """根据窗口大小缩放图像"""
        _fit_ratio = min(self._window_size[0] / arr.shape[0], self._window_size[1] / arr.shape[1])
        return Image.fromarray(arr).resize([int(arr.shape[1] * _fit_ratio), int(arr.shape[0] * _fit_ratio)],
                                           Image.BILINEAR)

    def prev_page_callback(self, *args):
        """上一张图像回调函数"""
        if self.clock.tick() is False:
            return
        if self.current_index <= 0:
            self.current_index = self.total_img_num - 1
        else:
            self.current_index -= 1
        self.load_images()

    def next_page_callback(self, *args):
        """下一张图像回调函数"""
        if self.clock.tick() is False:
            return
        if self.current_index >= self.total_img_num - 1:
            self.current_index = 0
        else:
            self.current_index += 1
        self.load_images()

    def close_window_callback(self):
        """关闭子窗口时，绑定在子类实例上的数组所占内存并没有被释放，容易导致内存溢出。
        因此自定义关闭窗口回调函数，释放其所占内存"""
        del self.hu_arrs
        del self.suv_arrs
        self.top_level.destroy()
