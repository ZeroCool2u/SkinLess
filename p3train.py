import hpelm
from sys import argv
from datetime import datetime
import gc

__project__ = "ELMTrainSlave3"
__author__ = "Theo Linnemann"

ORIGINAL_SUBFOLDERS = [
    ('/Shared/bdagroup3/FaceSkinDataset/Original/train', '/Shared/bdagroup3/FaceSkinDataset/XTrainUJ.h5'),
    ('/Shared/bdagroup3/FaceSkinDataset/Original/test', '/Shared/bdagroup3/FaceSkinDataset/XTestUJ.h5')]

SKIN_SUBFOLDERS = [('/Shared/bdagroup3/FaceSkinDataset/Skin/train', '/Shared/bdagroup3/FaceSkinDataset/TTrainUJ.h5'),
                   ('/Shared/bdagroup3/FaceSkinDataset/Skin/test', '/Shared/bdagroup3/FaceSkinDataset/TTestUJ.h5')]


def addData(dimensions, startRow, numRows):
    model0 = hpelm.HPELM(dimensions[0], dimensions[1])
    model0.load('/Shared/bdagroup3/pmodelMaster1500.hf')
    model0.add_data('/Shared/bdagroup3/FaceSkinDataset2/XTrainUJ.h5', '/Shared/bdagroup3/FaceSkinDataset2/TTrainUJ.h5',
                    istart=startRow, icount=numRows, fHH="/Shared/bdagroup3/DONOTTOUCH/HH.h5",
                    fHT="/Shared/bdagroup3/DONOTTOUCH/HT.h5")


def main(argv):
    dimensions = (75, 75)
    startRow = 3000000
    numRows = 1000000
    print("Starting Row: " + str(startRow))
    print("Assigned Number of Rows: " + str(numRows))
    print("Adding data now.")
    addData(dimensions, startRow, numRows)
    print("Adding data complete.")
    gc.collect()


if __name__ == "__main__":
    run_date = datetime.now()
    print('Running...', __project__, 'by', __author__, 'on', run_date.strftime("%m-%d-%Y"))
    print(' ')

    main(argv[1:])

    print(' ')
    print('Done with', __project__)
