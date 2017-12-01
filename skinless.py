from hpelm import modules as mod
import glob
import numpy as np
from PIL import Image
import os
import math

def fullSet():
    #TODO@Naren: Finish this function plz.
    pass

def unionJackPrep(DATALOCATION:str , SUBSET:str )->list:
    m = []
    all_files = []
    print("Current Subset: " + SUBSET)
    for filename in os.listdir( os.path.join(DATALOCATION, SUBSET) ):
        all_files.append(filename)
    all_files.sort()
    for q in range(0, len(all_files)):
        print("Current file: " + all_files[q])
        im = Image.open(os.path.join(os.path.join(DATALOCATION, SUBSET), all_files[q]))
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
                            a.append(r)
                            a.append(g)
                            a.append(b)
                m.append(a)
                q = q + d
            p = p + d
            q = 0
    return m

LOCAL = True
d = 7

if LOCAL:
    DATALOCATION = "/Users/macbook_user/Projects/SkinLess/FaceSkinDataset"
else:
    DATALOCATION = "/Shared/bdagroup3/FaceSkinDataset"

SUBFOLDERS = ['Original/train', 'Original/test/', 'Skin/train/', 'Skin/test/']

matrix = unionJackPrep(DATALOCATION, SUBFOLDERS[0])

for row in matrix:
    print("Row Size: " + str(len(row)))

print("Total matrix size: " + str(len(matrix)))
print("")
#
# OriginalTrain = glob.glob(os.path.join(DATALOCATION, '/Original/train/*.jpg'))
# print("Completed reading in OriginalTrain.")
# OriginalTest = glob.glob(os.path.join(DATALOCATION, '/Original/test/*.jpg'))
# print("Completed reading in OriginalTest.")
# SkinTrain = glob.glob(os.path.join(DATALOCATION, '/Skin/train/*.bmp'))
# print("Completed reading in SkinTrain.")
# SkinTest = glob.glob(os.path.join(DATALOCATION, '/Skin/test/*.bmp'))
# print("Completed reading in SkinTest.")

# npOrgTrain = np.array([np.array(Image.open(fname)) for fname in OriginalTrain])
# print("Completed np conversion for OrgTrain.")
# nporgTest = np.array([np.array(Image.open(fname)) for fname in OriginalTest])
# npSkinTrain = np.array([np.array(Image.open(fname)) for fname in SkinTrain])
# print("Completed np conversion for SkinTrain.")
# npSkinTest = np.array([np.array(Image.open(fname)) for fname in SkinTest])

# print("Starting HDF5 conversion for OrgTrain.")
# hdOrgTrain = mod.make_hdf5(npOrgTrain, '/Shared/bdagroup3/FaceSkinDataset/Original/train/hdOrgTrain.h5', dtype=object)
# print("Completed HDF5 conversion for OrgTrain.")
# print("Starting HDF5 conversion for SkinTrain.")
# hdSkinTrain = mod.make_hdf5(npSkinTrain, '/Shared/bdagroup3/FaceSkinDataset/Skin/train/hdSkinTrain.h5', dtype=object)
# print("Completed HDF5 conversion for SkinTrain.")



