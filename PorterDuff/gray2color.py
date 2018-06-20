import numpy as np
from matplotlib import cm


def _get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))
    return colormap_int


def gray2color(gray_array):
    color_map = _get_jet()
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)
    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]
    return color_array
