import hpelm
from sys import argv
from datetime import datetime
import gc

__project__ = "ELMTrainSlave6"
__author__ = "Theo Linnemann"

ORIGINAL_SUBFOLDERS = [
    ('/Shared/bdagroup3/FaceSkinDataset/Original/train', '/Shared/bdagroup3/FaceSkinDataset/XTrainUJ.h5'),
    ('/Shared/bdagroup3/FaceSkinDataset/Original/test', '/Shared/bdagroup3/FaceSkinDataset/XTestUJ.h5')]

SKIN_SUBFOLDERS = [('/Shared/bdagroup3/FaceSkinDataset/Skin/train', '/Shared/bdagroup3/FaceSkinDataset/TTrainUJ.h5'),
                   ('/Shared/bdagroup3/FaceSkinDataset/Skin/test', '/Shared/bdagroup3/FaceSkinDataset/TTestUJ.h5')]


def takeMeThurrr(dimensions):
    model0 = hpelm.HPELM(dimensions[0], dimensions[1])
    model0.load('/Shared/bdagroup3/pmodelMaster1500.hf')
    model0.solve_corr("/Shared/bdagroup3/DONOTTOUCH/HH.h5", "/Shared/bdagroup3/DONOTTOUCH/HT.h5")
    print("Solution found.")
    model0.save('/Shared/bdagroup3/pmodelMaster1500.hf')
    print("Model saved.")
    finalModel = hpelm.HPELM(dimensions[0], dimensions[1])
    finalModel.load('/Shared/bdagroup3/pmodelMaster1500.hf')
    finalModel.predict('/Shared/bdagroup3/FaceSkinDataset2/XTrainUJ.h5',
                       '/Shared/bdagroup3/FaceSkinDataset2/pYTrainUJ.h5')
    err_train = finalModel.error("/Shared/bdagroup3/FaceSkinDataset2/pYTrainUJ.h5",
                                 '/Shared/bdagroup3/FaceSkinDataset2/TTrainUJ.h5')
    print('Classification Training Error: ' + str(err_train))

    finalModel.predict('/Shared/bdagroup3/FaceSkinDataset2/XTestUJ.h5',
                       '/Shared/bdagroup3/FaceSkinDataset2/pYTestUJ.h5')
    err_test = finalModel.error("/Shared/bdagroup3/FaceSkinDataset2/pYTestUJ.h5",
                                '/Shared/bdagroup3/FaceSkinDataset2/TTestUJ.h5')
    print('Classification Test Error: ' + str(err_test))


def main(argv):
    dimensions = (75, 75)
    print("Solving ELM with given fHH (Covariance) & fHT (Correlation) matrix files.")
    takeMeThurrr(dimensions)
    gc.collect()


if __name__ == "__main__":
    run_date = datetime.now()
    print('Running...', __project__, 'by', __author__, 'on', run_date.strftime("%m-%d-%Y"))
    print(' ')

    main(argv[1:])

    print(' ')
    print('Done with', __project__)
