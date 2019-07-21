import os

import PIL import Image

def read_imgfile(root, path, name, width=None, height=None):
    val_image = Image.open(os.path.join(root, path, name))
    return val_image