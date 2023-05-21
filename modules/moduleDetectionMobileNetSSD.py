'''
file contains first implementation of object detection method. 
Primarily based on a pre-trained model utilizing Mobilenet-SSD, a single stage network pre-trained on several datasets 
'''

# --- / 
# -- / external imports 
import numpy as np 
import cv2 as opencv
from modules.moduleFileManagement import gatherFilePath

#import os

# --- / 
# -- / internal imports 
from settings.modelSettings import MSSDResizeFactor,MSSDDisplayOpacity

# --- / 
# -- / 

# --- / 
# -- / necessities for given image : 
# image file itself --> path 
# directory leading to YOLO instance --> pre-trained 
# confidence level to deploy 0.4 for at least 40% confidence / 
# threshold for non-maxima suppression // 

# --- / 
# -- / 
def generateLabelColors(len:int):
    '''
    returns List of len:int containing colors in format (r,g,b)
    - where r,g,b in range of (0 - 255) 
    - 
    ## example usage: 
    to obtain 2 colors represented in form of [(R,G,B),(R1,G1,B2,)]:
    generateLabelColors(2) -> will return np.array type
    '''
    # TODO better returntype!
    listOfColors = np.random.uniform(0,255, size=(len,3)   )# creating n 3-tuples of random colors 
    return listOfColors

# --- / 
# -- / 


def loadModel(modelPath:str,modelWeight:str) -> object:
    ''' 
    with given paths load and return neural net with cv2
    @param modelPath String = contains path to model in txt form 
    @param modelWeight String = contains path to pretrained weight of network 
    @return dnn Object containing loaded network
    ''' 
    # TODO exeption handling
    dnnModel = opencv.dnn.readNetFromCaffe(modelPath,modelWeight);
    print("model loaded")
    return dnnModel

# --- / 
# -- / reading image with opencv

def LoadImage(imagePath:str):
    '''
    extracting image from given path 
    '''
    #TODO add exception handling 
    try:
        Img = opencv.imread(imagePath)
    except: 
        exit("image not found, sorry")
    return Img 
    
# --- / 
# -- / processing image to blob

def ProcessImage(imagePath:str): 
    '''
    function loading and preparing an image for processing in network
    returns imageBlob -> processed by opencv 
    '''
    ImgConverted = LoadImage(imagePath);
    
    # require fixed size of image for MobileNet to process it accordingly
    ImgFrame = opencv.resize(ImgConverted,(MSSDResizeFactor,MSSDResizeFactor))
    # creation of blob --> processed image with normalized data 
    # resizing it and maintaining images by applying a _mean subtraction_
    Imgblob = opencv.dnn.blobFromImage(ImgFrame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    return Imgblob
# --- / 
# -- / gathering dimensions of supplied image 

def gatherImageDimensions(imagePath:str) :
    readImg = LoadImage(imagePath)
    (h,w) = readImg.shape[:2]
    return (h,w)    

# --- / 
# -- / running Dnn 

def runDnn(imagePath,neuralNet:object,requiredConfidence:float):
    '''
    takes an imagepath, neural network object - initialized, and the required confidence to achieve for detected objects. 
    
    
    
    #### run examples :
    loadedNet = MSSD.loadModel(MSSDnetwork,MSSDWeight)
    MSSD.runDnn(gatherFilePath("**/unsplashHis.jpg"),loadedNet,0.1)
    '''    
    #TODO Refactor into smaller portions 
    #TODO adapt for use with Classing
    # gathering image data 
    ImgQueried =LoadImage(imagePath)
    ImgResized = opencv.resize(ImgQueried,(MSSDResizeFactor,MSSDResizeFactor))
    imageBlob = ProcessImage(imagePath)
    
    # creating copies of image for displaying afterwards 
    imageCopy = ImgQueried.copy()
    imageCopy2 = ImgQueried.copy() # why two ?
    # setting scaling for better positioning of rectangles 
    imgWidthFactor = ImgQueried.shape[1]/MSSDResizeFactor
    imgHeightFactor = ImgQueried.shape[0]/MSSDResizeFactor
    imgColumns:int = MSSDResizeFactor
    imgRows:int = MSSDResizeFactor
     
    neuralNet.setInput(imageBlob)
    
    # looping over everything detected: 
    detectedObjects = neuralNet.forward()
    
    #TODO extract to function ProcessdetectedObjects() or similar 
    for prediction in range(detectedObjects.shape[2]):
        
        # extract confidence of prediction 
        confidence = detectedObjects[0,0,prediction,2];
        
        # check whether confidence reaches set level or not 
        if confidence > requiredConfidence: 
            # extracting label of detected image
            
            labelIndex = int(detectedObjects[0,0,prediction,1])
        
            #TODO refactor to displayRectangle()
            # location of object in coordinates
            cornerLeftX =int(int(detectedObjects[0, 0, prediction, 3] * imgColumns) * imgWidthFactor)
            cornerLeftY =int( int(detectedObjects[0, 0, prediction, 4] * imgRows) * imgHeightFactor)
            cornerRightX = int(int(detectedObjects[0, 0, prediction, 5] * imgColumns) * imgWidthFactor)
            cornerRightY= int(int(detectedObjects[0, 0, prediction, 6] * imgRows) * imgHeightFactor)

           
            # creating rectangles to poistion around 
            opencv.rectangle( ImgResized, (cornerLeftX,cornerLeftY) , (cornerRightX,cornerRightY),(0,255,0))
            opencv.rectangle( imageCopy2, (cornerLeftX,cornerLeftY) , (cornerRightX,cornerRightY),(0,255,0),-1)
    
    opencv.addWeighted(imageCopy, MSSDDisplayOpacity, ImgQueried , 1 - MSSDDisplayOpacity, 0, ImgQueried)
    
    # second loop for improved imaging / search 
    for prediction in range(detectedObjects.shape[2]): 
        
        # gathering confidence of found object
        confidence = detectedObjects[0,0,prediction,2]
        
        if confidence > requiredConfidence: 
            # extracting label of detected image
            
            labelIndex = int(detectedObjects[0,0,prediction,1])
        
            # location of object in coordinates# location of object in coordinates
            cornerLeftX =int(int(detectedObjects[0, 0, prediction, 3] * imgColumns) * imgWidthFactor)
            cornerLeftY =int( int(detectedObjects[0, 0, prediction, 4] * imgRows) * imgHeightFactor)
            cornerRightX = int(int(detectedObjects[0, 0, prediction, 5] * imgColumns) * imgWidthFactor)
            cornerRightY= int(int(detectedObjects[0, 0, prediction, 6] * imgRows) * imgHeightFactor)
            
            # creating rectangles to poistion around 
            opencv.rectangle( ImgQueried, (cornerLeftX,cornerLeftY) , (cornerRightX,cornerRightY),(0,0,0),2)
            # we now have to set and enable the given label and rectangle to be positioned and shown on our implementation
            if labelIndex in CLASSES:
                # found label, now creating according text-outputs 
                label = "{} => {}".format(CLASSES[labelIndex],str(confidence))
                print("prediction : {}".format(label))
                
                labelSize, maxLabelHeight = opencv.getTextSize(label, opencv.FONT_HERSHEY_TRIPLEX, 0.8, 1)
                
                # gathering point for positioning label accordingly
                # deciding whether to take leftBottom corner for reference or label
                cornerLeftY = max(cornerLeftY,labelSize[1])
                # setting labelbox rectangle coordinates accordingly
                # reference is drawn from bottom left corner 
                LabelLeftY = cornerLeftY - labelSize[1]
                LabelRightY = cornerLeftY + maxLabelHeight 
                LabelRightX = cornerLeftX + labelSize[0]
                # drawing LabelBox 
                opencv.rectangle(ImgQueried,(cornerLeftX,LabelLeftY), (LabelRightX,LabelRightY),(0,0,0))
                # setting Text into Box
                opencv.putText(ImgQueried,label, (cornerLeftX,LabelLeftY), opencv.FONT_HERSHEY_TRIPLEX,0.8,(200,200,200))
                
                
                
    # display the prediction       
    opencv.namedWindow("frame",opencv.WINDOW_NORMAL)
    opencv.imshow("frame",ImgQueried)
    opencv.waitKey(0)
    opencv.destroyAllWindows()

# --- / 
# -- / 

CLASSES:dict = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

COLORLABEL =generateLabelColors(len(CLASSES))
# here static, in final product it should be constructed from streamlit checklist! 
# constructing interface which will then supply the selected classes 
# def SupplyClasses

  
    
if __name__ =="__main__":
    exit("not meant to be run")