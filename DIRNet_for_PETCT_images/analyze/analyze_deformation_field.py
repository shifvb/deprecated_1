import os
import pickle
import numpy as np


def main():
    work_dir = r'F:\registration_running_data\def_vec'
    filenames = [os.path.join(work_dir, _) for _ in os.listdir(work_dir)]
    for filename in filenames:
        obj = pickle.load(open(filename, 'rb'))
        obj_x = obj[0, :, :, 0]
        obj_x = obj_x * 255
        obj_y = obj[0, :, :, 1]
        pass

if __name__ == '__main__':
    main()
