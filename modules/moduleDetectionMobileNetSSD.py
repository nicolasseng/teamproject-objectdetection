'''
file contains first implementation of object detection method. 
Primarily based on a pre-trained model utilizing Mobilenet-SSD, a single stage network pre-trained on several datasets 
'''

# --- / 
# -- / external imports 
from typing import Optional
from altair import Undefined
import numpy as np 
import cv2 as opencv
from streamlit.elements.color_picker import ColorPickerMixin
from modules.moduleFileManagement import gatherFilePath

#import os

# --- / 
# -- / internal imports 
from settings.modelSettings import MSSDResizeFactor,MSSDDisplayOpacity,ImageTextColor, RectangleWidth


# --- / 
# -- / 
def generateLabelColors(len:int) -> list:
    '''
    returns List of len:int containing colors in format (r,g,b)
    - where r,g,b in range of (0 - 255) 
    - 
    ## example usage: 
    to obtain 2 colors represented in form of [(R,G,B),(R1,G1,B2,)]:
    generateLabelColors(2) -> will return np.array type
    '''
    listOfColor:list = [ [ round(np.random.uniform(0,255),2) for i in range(0,3) ] for i in range(0,len) ]
    return listOfColor

# --- / 
# -- / 

def loadModel(modelPath:str,modelWeight:str) -> object:
    ''' 
    with given paths load and return neural net with cv2
    **raises Exception in case of failed loading operation**
    @param modelPath String = contains path to model in txt form 
    @param modelWeight String = contains path to pretrained weight of network 
    @return dnn Object containing loaded network
    ''' 
    try:
        dnnModel = opencv.dnn.readNetFromCaffe(modelPath,modelWeight);
    except: 
        raise Exception("Model could not be load!")
    return dnnModel

# --- / 
# -- / reading image with opencv

def LoadImage(imagePath:str):
    '''
    extracting image from given path 
    '''
    try:
        Img = opencv.imread(imagePath)
    except: 
        exit("image not found, sorry")
    return Img 
    
# --- / 
# -- / processing image to blob

def ProcessImage(imageObject:object): 
    '''
    function loading and preparing an image for processing in network
    returns imageBlob -> processed by opencv 
    '''
    ImgConverted = imageObject
    # ImgConverted = LoadImage(imagePath);
    
    # require fixed size of image for MobileNet to process it accordingly
    ImgFrame = opencv.resize(ImgConverted,(MSSDResizeFactor,MSSDResizeFactor))
    # creation of blob --> processed image with normalized data 
    # resizing it and maintaining images by applying a _mean subtraction_
    Imgblob = opencv.dnn.blobFromImage(ImgFrame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    return Imgblob
        
# --- / 
# -- / wrapper for Dnn 

def wrapperRunningDnn(neuralNet:object,requiredConfidence:float,streamlitOutput,imageObj=Optional,videoStream=Optional,suppliedObjectClasses:dict=Optional):
    ''' 
    creates wrapper for "runDnn"
    @param neuralNet = initialized neural network object 
    @param requiredConfidence:float = float defining required confidence 
    @param imageObj = imageObject previously converted to numpy.array (by streamlit or similar application)
    '''
    
    
    # setting default value for objectClasses, if they are not set
    if suppliedObjectClasses is wrapperRunningDnn.__defaults__[2]:
        objectClasses:dict = { 0: 'background',
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
        14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
    else: 
        objectClasses:dict = suppliedObjectClasses
    
    if videoStream is not wrapperRunningDnn.__defaults__[1] : 
        ObjectColors = generateLabelColors(len(objectClasses))
        # running in continous loop for capturing images and processing them 
        #TODO implement better stopping --> shouldnt be a problem because streamlit reruns the whole python script anyway! 
        while True:
            result,frame = videoStream.read()
            DetectionResults = runDnn(frame,neuralNet,requiredConfidence,objectClasses,ObjectColors)
            
            # correcting color mismatch after processing with opencv
            resultedImage = opencv.cvtColor(DetectionResults['image'],opencv.COLOR_BGR2RGB)
            streamlitOutput.image(resultedImage)
    
    if imageObj is not  wrapperRunningDnn.__defaults__[0]: 
        # running Dnn on supplied image
        DetectionResults = runDnn(imageObj,neuralNet,requiredConfidence,objectClasses)
        streamlitOutput.image(DetectionResults['image'],
                                use_column_width="auto",
                                caption="Image  object detection",
                              )


# --- / 
# -- / running Dnn 

def runDnn(imageObject,neuralNet:object,requiredConfidence:float,objectClasses:dict,ClassColor=None) -> dict:
    '''
    takes an imageObject, neural network object - initialized, and the required confidence to achieve for detected objects. 
    
    #### run examples :
    loadedNet = MSSD.loadModel(MSSDnetwork,MSSDWeight)
    MSSD.runDnn(gatherFilePath("**/unsplashHis.jpg"),loadedNet,0.1)
    '''    
    if ClassColor == None : 
        ClassColor = generateLabelColors(len(objectClasses))
    
    # gathering image data 
    imageBlob = ProcessImage(imageObject) 
    
    # pipe imageblob to neural net
    neuralNet.setInput(imageBlob)
    detectedObjects = neuralNet.forward()
    
    # iterate through all results
    DetectionResults:dict = iterateDetectionResults(imageObject,
                            detectedObjects,
                            requiredConfidence,
                            objectClasses,
                            ClassColor
                            )
    
    return DetectionResults
    

# --- / 
# -- / 
def addRectangle(imageObj,BoundingBoxCoords:dict, color):

    # calculating position of rectangle corners
    return opencv.rectangle( imageObj,
                            (BoundingBoxCoords['xLeft'],
                            BoundingBoxCoords['yLeft']),
                            (BoundingBoxCoords['xRight'],
                            BoundingBoxCoords['yRight']),
                            color,RectangleWidth)
    
# --- /
# -- / 
def addLabel(imageObj,detectionBoundingBox:dict,Label:str,confidence:float,color):
    
    textLabel = "{}:{}".format(Label,str(confidence))
    # print("prediction : {}".format(label))
    
    labelSize, maxLabelHeight = opencv.getTextSize(textLabel, opencv.FONT_HERSHEY_TRIPLEX, 1, 1)
    
    # defining coordinates for  
    LabelBoxY = detectionBoundingBox['yLeft']- labelSize[1]
    LabelLeftY = detectionBoundingBox['yLeft'] 
    LabelRightX = detectionBoundingBox['xLeft']+ labelSize[0]
    
    # drawing rectangle around text 
    opencv.rectangle(imageObj,(detectionBoundingBox['xLeft'],LabelBoxY), (LabelRightX,LabelLeftY),color,RectangleWidth)
    
    # drawing text above detected object 
    opencv.putText(imageObj,textLabel, (detectionBoundingBox['xLeft'],LabelLeftY), opencv.FONT_HERSHEY_TRIPLEX,1,ImageTextColor,1,opencv.LINE_8,False)
    
    return imageObj
 
# --- /
# -- / 
def iterateDetectionResults(imageObj,detectedObjects,requiredConfidence,objectClasses,ObjectClassColor):
    
    # list containing dictionary with their name : color
    foundObjects:dict = {}
    
    for prediction in range(detectedObjects.shape[2]):
        
        # extract confidence of prediction 
        rawconfidence = detectedObjects[0,0,prediction,2];
        confidence = round(rawconfidence,3)
        
        if confidence > requiredConfidence: 
            # extracting label of detected image
            labelIndex = int(detectedObjects[0,0,prediction,1])
            
            imgWidthFactor = imageObj.shape[1]/MSSDResizeFactor
            imgHeightFactor = imageObj.shape[0]/MSSDResizeFactor
            detectionBoundingBox:dict = {
                "xLeft": int(int( detectedObjects[0,0,prediction,3] * MSSDResizeFactor) * imgWidthFactor ) ,
                "yLeft":int(int(detectedObjects[0,0,prediction,4] * MSSDResizeFactor) * imgHeightFactor  ),
                "xRight":int(int(detectedObjects[0,0,prediction,5] * MSSDResizeFactor) * imgWidthFactor ),
                "yRight":int(int(detectedObjects[0,0,prediction,6] * MSSDResizeFactor) * imgHeightFactor ),
            }
            
            # setting rectangle about found object 
            imageWithRect = addRectangle(imageObj,
                                         detectionBoundingBox,
                                         color=ObjectClassColor[labelIndex]
                                         )
                        
            if labelIndex in objectClasses:
                
                # found label, now creating according text-outputs 
                ObjectClassString = objectClasses[labelIndex]
                
                # adding entry to dictionary to return! 
                foundObjects[ObjectClassString] = ObjectClassColor[labelIndex] 
                imageObj = addLabel(imageWithRect,detectionBoundingBox,ObjectClassString,confidence,ObjectClassColor[labelIndex])
    
    Results:dict= {
        "image": imageObj,
        "foundObjects": foundObjects,
    }
    return  Results

if __name__ =="__main__":
    exit("not meant to be run")