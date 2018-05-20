import os
import shutil
from PIL import Image
movimgs = []
fiximgs= []
rstimgs = []
for i in range(10):
    _path = os.path.join(r"F:\registration_patches\mnist\test", str(i))
    _files = os.listdir(_path)
    movimgs.extend([os.path.join(_path, _) for _ in _files if "x" in _])
    fiximgs.extend([os.path.join(_path, _) for _ in _files if "y" in _])
    rstimgs.extend([os.path.join(_path, _) for _ in _files if "z" in _])

movimg_folder = r"F:\registration_patches\mnist\test\moving_images"
fiximg_folder = r"F:\registration_patches\mnist\test\fixed_images"
if not os.path.exists(movimg_folder):
    os.mkdir(movimg_folder)
if not os.path.exists(fiximg_folder):
    os.mkdir(fiximg_folder)
for i in range(len(movimgs)):
    _name = "{:>05}.png".format(i)
    img = Image.open(movimgs[i])
    img.save(os.path.join(movimg_folder, _name))
    img = Image.open(fiximgs[i])
    img.save(os.path.join(os.path.join(fiximg_folder, _name)))

