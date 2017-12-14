import os
import hpelm
from hpelm import modules
from sys import argv
from datetime import datetime
import gc

__project__ = "SkinLess"
__author__ = "Theo Linnemann"

def setupHPELM():
    model0 = hpelm.HPELM(75, 75, precision='double', classification='c', tprint=30)
    model0.add_neurons(1000, 'sigm')
    model0.save('/Shared/bdagroup3/model1000.hf')
    return model0


def main(argv):
    model = setupHPELM()
    model.train('/Shared/bdagroup3/FaceSkinDataset2/XTrainUJ.h5', '/Shared/bdagroup3/FaceSkinDataset2/TTrainUJ.h5')
    model.predict('/Shared/bdagroup3/FaceSkinDataset2/XTrainUJ.h5', '/Shared/bdagroup3/FaceSkinDataset2/YTrainUJ.h5')
    err_train = model.error("/Shared/bdagroup3/FaceSkinDataset2/YTrainUJ.h5",
                            '/Shared/bdagroup3/FaceSkinDataset2/TTrainUJ.h5')
    print('Classification Training Error: ' + str(err_train))

    model.predict('/Shared/bdagroup3/FaceSkinDataset2/XTestUJ.h5', '/Shared/bdagroup3/FaceSkinDataset2/YTestUJ.h5')
    err_test = model.error("/Shared/bdagroup3/FaceSkinDataset2/YTestUJ.h5",
                           '/Shared/bdagroup3/FaceSkinDataset2/TTestUJ.h5')
    print('Classification Test Error: ' + str(err_test))

if __name__ == "__main__":
    run_date = datetime.now()
    print('Running...', __project__, 'by', __author__, 'on', run_date.strftime("%m-%d-%Y"))
    print(' ')

    main(argv[1:])

    print(' ')
    print('Done with', __project__)
