import settings.modelSettings as msettings
import modules.moduleDetectionMobileNetSSD as MSSD
from settings.modelSettings import MSSDnetwork, MSSDWeight,gatherFilePath


loadedNet = MSSD.loadModel(MSSDnetwork,MSSDWeight)
print(type(MSSD.wrapperRunningDnn(loadedNet,0.1,gatherFilePath("**/sampleImg.jpg"),)))
