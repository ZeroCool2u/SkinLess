import math
import os
import glob
from sys import argv
from datetime import datetime
from PIL import Image
from tables import open_file, Atom, Filters
import numpy as np
import gc
import sys

__project__ = "MaskGen"
__author__ = "Theo Linnemann"
LOCAL = False
d = 7

ORIGINAL_SUBFOLDERS = [
    ('/Shared/bdagroup3/FaceSkinDataset2/Original/train', '/Shared/bdagroup3/FaceSkinDataset2/XTrainUJ.h5'),
    ('/Shared/bdagroup3/FaceSkinDataset2/Original/test', '/Shared/bdagroup3/FaceSkinDataset2/XTestUJ.h5')]
# ('/Shared/bdagroup3/FaceSkinDataset/Original/val', '/Shared/bdagroup3/FaceSkinDataset/orgVal.hdf5')]

SKIN_SUBFOLDERS = [('/Shared/bdagroup3/FaceSkinDataset2/Skin/train', '/Shared/bdagroup3/FaceSkinDataset2/TTrainUJ.h5'),
                   ('/Shared/bdagroup3/FaceSkinDataset2/Skin/test', '/Shared/bdagroup3/FaceSkinDataset2/TTestUJ.h5')]


#                       ('/Shared/bdagroup3/FaceSkinDataset/Skin/val', '/Shared/bdagroup3/FaceSkinDataset/skinVal.hdf5')]

def fullDataPrep(SUBSET):
    m = []
    all_files = glob.glob(os.path.join(SUBSET, r'*.*'))
    print("Current Subset: " + SUBSET)
    all_files.sort()
    # Force ALL the image handles to be loaded into memory now.
    images = list(map(Image.open, all_files))
    for q in range(0, len(images)):
        print("Current file: " + all_files[q])
        im = images[q]
        width, height = im.size
        width_blocks = int(math.floor(width / d))
        height_blocks = int(math.floor(height / d))
        rgb_im = im.convert('RGB')
        p = q = s = 0
        for i in range(0, height_blocks):
            for j in range(0, width_blocks):
                a = []
                for k in range(0, d):
                    for l in range(0, d):
                        r, g, b = rgb_im.getpixel((q + k, p + l))
                        if (r != 255):
                            r = 1
                            a.append(r)
                        else:
                            r = 0
                            a.append(r)
                        if (g != 255):
                            g = 1
                            a.append(g)
                        else:
                            g = 0
                            a.append(g)
                        if (b != 255):
                            b = 1
                            a.append(b)
                        else:
                            b = 0
                            a.append(b)
                m.append(a)
                q = q + d
            p = p + d
            q = 0
    return np.asarray(m, dtype=np.float64)


def unionJackPrep(SUBSET):
    m = []
    all_files = glob.glob(os.path.join(SUBSET, r'*.*'))
    print("Current Subset: " + SUBSET)
    all_files.sort()
    # Force ALL the image handles to be loaded into memory now.
    images = list(map(Image.open, all_files))
    for q in range(0, len(images)):
        print("Current file: " + all_files[q])
        im = images[q]
        width, height = im.size
        width_blocks = int(math.floor(width / d))
        height_blocks = int(math.floor(height / d))
        rgb_im = im.convert('RGB')
        p = q = s = 0
        for i in range(0, height_blocks):
            for j in range(0, width_blocks):
                a = []
                for k in range(0, d):
                    for l in range(0, d):
                        if (k == l) or (k == math.floor(d / 2)) or (l == math.floor(d / 2)) or (k + l == d - 1):
                            r, g, b = rgb_im.getpixel((q + k, p + l))
                            if (r != 255):
                                r = 1
                                a.append(r)
                            else:
                                r = 0
                                a.append(r)
                            if (g != 255):
                                g = 1
                                a.append(g)
                            else:
                                g = 0
                                a.append(g)
                            if (b != 255):
                                b = 1
                                a.append(b)
                            else:
                                b = 0
                                a.append(b)
                m.append(a)
                q = q + d
            p = p + d
            q = 0
    return np.asarray(m, dtype=np.float64)


def main(argv):
    # Manually change the list we iterate through to select between the data and masks. (Doing both kills the node.)
    for file, saveTarget in SKIN_SUBFOLDERS:
        h5file = saveTarget
        h5 = open_file(h5file, "w")
        X = unionJackPrep(file)
        atom = Atom.from_dtype(X.dtype)
        flt = Filters(complevel=0)
        h5data = h5.create_carray(h5.root, "data", atom, X.shape, filters=flt)
        h5data[:] = X
        h5data.attrs.mean = None
        h5data.attrs.std = None
        h5.flush()
        h5.close()
        del h5
        del X
        del atom
        del flt
        del h5data
        gc.collect()
        print("No mean or std to compute.")


if __name__ == "__main__":
    run_date = datetime.now()
    print('Running...', __project__, 'by', __author__, 'on', run_date.strftime("%m-%d-%Y"))
    print(' ')

    main(argv[1:])

    print(' ')
    print('Done with', __project__)
    sys.exit(0)
