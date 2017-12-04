import os
import hpelm
from hpelm import modules
from sys import argv
from datetime import datetime
import gc

__project__ = "SkinLess"
__author__ = "Theo Linnemann"

ORIGINAL_SUBFOLDERS = [
    ('/Shared/bdagroup3/FaceSkinDataset/Original/train', '/Shared/bdagroup3/FaceSkinDataset/XTrainUJ.h5'),
    ('/Shared/bdagroup3/FaceSkinDataset/Original/test', '/Shared/bdagroup3/FaceSkinDataset/XTestUJ.h5')]

SKIN_SUBFOLDERS = [('/Shared/bdagroup3/FaceSkinDataset/Skin/train', '/Shared/bdagroup3/FaceSkinDataset/TTrainUJ.h5'),
                   ('/Shared/bdagroup3/FaceSkinDataset/Skin/test', '/Shared/bdagroup3/FaceSkinDataset/TTestUJ.h5')]


def setupHPELM():
    model0 = hpelm.HPELM(75, 75, precision='double', classification='c', tprint=30)
    model0.add_neurons(3000, 'sigm')
    model0.save('/Shared/bdagroup3/model3000.hf')
    return model0


def main(argv):
    model = setupHPELM()
    model.train('/Shared/bdagroup3/FaceSkinDataset/XTrainUJ.h5', '/Shared/bdagroup3/FaceSkinDataset/TTrainUJ.h5')
    model.predict('/Shared/bdagroup3/FaceSkinDataset/XTrainUJ.h5', '/Shared/bdagroup3/FaceSkinDataset/YUJ.h5')
    err_train = model.error("/Shared/bdagroup3/FaceSkinDataset/XTrainUJ.h5", "/Shared/bdagroup3/FaceSkinDataset/YUJ.h5")
    print('Training Error: ' + str(err_train))

if __name__ == "__main__":
    run_date = datetime.now()
    print('Running...', __project__, 'by', __author__, 'on', run_date.strftime("%m-%d-%Y"))
    print(' ')

    main(argv[1:])

    print(' ')
    print('Done with', __project__)
