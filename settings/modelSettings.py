''' 
file contains variables and functions necessary for directing path to the correct directory.
For example "YoloWeightPath" will direc to the correct path containing the pre-trained model weighs necessary for it operating correctly 

'''
# --- /
# -- / external imports 

# --- / 
# -- / internal imports 
from modules.moduleFileManagement import gatherFilePath
# use moduleFileManagement.py 

MSSDWeight:str = gatherFilePath("**/MobileNetSSD_deploy.caffemodel")

MSSDnetwork:str = gatherFilePath("**/MobileNetSSD_deploy.prototxt")

# resizing factor for MobileNet Image Pre-Processing 
MSSDResizeFactor:int = 300

MSSDDisplayOpacity: float = 0.3

# -- / settings for Drawing Elements

RectangleWidth = 2
ImageTextColor = (0,255,0)


if __name__ =="__main__": 
    
    exit("not meant to be run")