import hpelm
from sys import argv
from datetime import datetime
import gc

__project__ = "ELMTrainMaster"
__author__ = "Theo Linnemann"

ORIGINAL_SUBFOLDERS = [
    ('/Shared/bdagroup3/FaceSkinDataset/Original/train', '/Shared/bdagroup3/FaceSkinDataset/XTrainUJ.h5'),
    ('/Shared/bdagroup3/FaceSkinDataset/Original/test', '/Shared/bdagroup3/FaceSkinDataset/XTestUJ.h5')]

SKIN_SUBFOLDERS = [('/Shared/bdagroup3/FaceSkinDataset/Skin/train', '/Shared/bdagroup3/FaceSkinDataset/TTrainUJ.h5'),
                   ('/Shared/bdagroup3/FaceSkinDataset/Skin/test', '/Shared/bdagroup3/FaceSkinDataset/TTestUJ.h5')]


def setupHPELM():
    # TODO: Complete parallel trial run.
    # TODO: Complete 3x3, 5x5, 9x9 preprocessing.
    # TODO: Ensemble the above models. 
    model0 = hpelm.HPELM(75, 75, precision='double', classification='c', tprint=30)
    model0.add_neurons(1500, 'sigm')
    model0.save('/Shared/bdagroup3/pmodelMaster1500.hf')


def main(argv):
    setupHPELM()
    print("Model write generation and write complete. Starting process pool...")
    gc.collect()


if __name__ == "__main__":
    run_date = datetime.now()
    print('Running...', __project__, 'by', __author__, 'on', run_date.strftime("%m-%d-%Y"))
    print(' ')

    main(argv[1:])

    print(' ')
    print('Done with', __project__)
