import os
import pickle
import numpy as np


def main():
    work_dir = r'F:\registration_running_data\def_vec'
    filenames = [os.path.join(work_dir, _) for _ in os.listdir(work_dir)]
    for filename in filenames:
        obj = pickle.load(open(filename, 'rb'))
        # obj = obj.reshape([32, 2, 128, 128]).transpose([0, 2, 3, 1])
        obj_x = obj[26, :, :, 1]
        obj_x = -(obj_x / 2 * 128).astype(np.int16)
        pass

if __name__ == '__main__':
    main()
