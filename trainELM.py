import os
import hpelm
from sys import argv
from datetime import datetime

__project__ = "SkinLess"
__author__ = "Theo Linnemann"

d = 7

ORIGINAL_SUBFOLDERS = [
    ('/Shared/bdagroup3/FaceSkinDataset/Original/train', '/Shared/bdagroup3/FaceSkinDataset/orgTrain.hdf5'),
    ('/Shared/bdagroup3/FaceSkinDataset/Original/test', '/Shared/bdagroup3/FaceSkinDataset/orgTest.hdf5')]

SKIN_SUBFOLDERS = [('/Shared/bdagroup3/FaceSkinDataset/Skin/train', '/Shared/bdagroup3/FaceSkinDataset/skinTrain.hdf5'),
                   ('/Shared/bdagroup3/FaceSkinDataset/Skin/test', '/Shared/bdagroup3/FaceSkinDataset/skinTest.hdf5')]


def setupHPELM():
    # TODO: Start here: https://hpelm.readthedocs.io/en/latest/parallel.html
    pass


def main(argv):
    HPELM = setupHPELM()


if __name__ == "__main__":
    run_date = datetime.now()
    print('Running...', __project__, 'by', __author__, 'on', run_date.strftime("%m-%d-%Y"))
    print(' ')

    main(argv[1:])

    print(' ')
    print('Done with', __project__)
