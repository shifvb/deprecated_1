import numpy as np

from matplotlib import cm


def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)

    for i in range(0, 256, 1):
        colormap_float[i, 0] = cm.jet(i)[0]
        colormap_float[i, 1] = cm.jet(i)[1]
        colormap_float[i, 2] = cm.jet(i)[2]

        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))

    # np.savetxt("jet_float.txt", colormap_float, fmt="%f", delimiter=' ', newline='\n')
    # np.savetxt("jet_int.txt", colormap_int, fmt="%d", delimiter=' ', newline='\n')

    # print(colormap_int)
    return colormap_int

_f = cm.cividis_r
def get_X():
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)


    for i in range(0, 256, 1):

        colormap_int[i, 0] = np.int_(np.round(_f(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(_f(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(_f(i)[2] * 255.0))

    # np.savetxt("spectral_float.txt", colormap_float, fmt="%f", delimiter=' ', newline='\n')
    # np.savetxt("spectral_int.txt", colormap_int, fmt="%d", delimiter=' ', newline='\n')

    return colormap_int


def gray2color(gray_array):
    color_map = get_jet()
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]

            # color_image = Image.fromarray(color_array)

    return color_array
