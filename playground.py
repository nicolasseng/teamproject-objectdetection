import settings.modelSettings as msettings
import modules.moduleDetectionMobileNetSSD as MSSD
from settings.modelSettings import MSSDnetwork, MSSDWeight,gatherFilePath


loadedNet = MSSD.loadModel(MSSDnetwork,MSSDWeight)

MSSD.runOnVideoStream(loadedNet,0.2)