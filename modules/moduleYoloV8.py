"""
file contains functions and structures necessary for running yolov8 or similar versions
with our object detection webapp. More specifically it is only supplying the logic 
necesssary and is later piped or accessed by the UI.  
"""

# --- /
# -- / external imports 
from typing import Callable, Optional
import cv2 
import time
from matplotlib.cm import np
from ultralytics import YOLO
# TODO to be removed once refactored! 
import streamlit as st
from modules.moduleDetectionMobileNetSSD import wrapperRunningDnn
# TODO remove those libraries because they are not needed here --> at least should not be 

# --- / 
# -- / internal imports 
from modules.moduleFileManagement import gatherFilePath, gatherFolderPath, createPath
from modules.moduleYamlManagement import createYaml

# --- / 
# -- / 
def initializeModel(selectedModel:str) -> object: 
    '''
    takes string to trained yolo detection model 
    returns loaded model for further use 
    
    ### sample usage  : 
    initializeModel("path/to/yolov8n.pt) -> returns accordingly loaded yolo detection model
    '''
    
    selectedModelPath:str = gatherFilePath("**/{}".format(selectedModel))

    if selectedModelPath == None : 
        # only occurs if our selected model was not downloaded yet 
        # TODO add automatic download to designated space in data/preTrainedYolo
        modelFolderPath:Optional[str] =gatherFolderPath("**/preTrainedYolo")
        selectedModel = createPath(modelFolderPath,selectedModel)
        print(selectedModel)
        return YOLO(selectedModel)
        
    # model:object = load_model(selectedModelPath)
    model:object = YOLO(selectedModelPath)
    return model
    

# --- / 
# -- / 
def modelOnVideo(functionToRun:Callable ,loadedModel,videoStream,streamlitOutput,streamlitFPSCounter,objectClasses,requiredConfidence) -> Optional[bool]:
    '''
    this function takes a videostream and processes its frames by running them on a loaded yoloV8 model 
    this videoStream can either be a **video** or **webcam input** 
    once it is done processing it will return a bool indicating that it stopped!  
    '''
    
    timeBeforeProcessing:float = 0 
    timeAfterProcessing:float = 0 
    
    while videoStream.isOpened():
        
        result, frame = videoStream.read() 
        
        if not result: # not parsing if nothing was received 
            return None # --> indicates issue!
        processedFrame = cv2.resize(frame, (720, int(720*(9/16))))
        # running detection on it 
        result = functionToRun(loadedModel,requiredConfidence, objectClasses,processedFrame) 
        #yoloOnImage(loadedModel,requiredConfidence, objectClasses,processedFrame)
        #mssdOnImage(loadedModel,requiredConfidence, objectClasses,processedFrame)
        
        timeAfterProcessing:float = time.time()
        
        #calculate fps 
        currentFPS:float = round(1/ (timeAfterProcessing - timeBeforeProcessing),2)
        
        # overwritting old value
        timeBeforeProcessing:float = timeAfterProcessing   
        
        # displaying resulted detection with StreamlitOutput!
        streamlitOutput.image(result['image'], caption="Detection in Video", use_column_width=True)
        # display fps 
        streamlitFPSCounter.markdown(("**{}**".format(currentFPS)))
    
    return True

# --- / 
# -- /
def yoloOnVideo(loadedModel,videoStream,streamlitOutput,streamlitFPSCounter,objectClasses,requiredConfidence) -> Optional[bool]:
    ''' 
    wrapper function for **YoloV8** detection, executing modelOnVideo with 
    predefined function for Yolo object detection
    '''
    modelOnVideo(runYoloOnImage,loadedModel,videoStream,streamlitOutput,streamlitFPSCounter,objectClasses,requiredConfidence)

# --- / 
# -- /
def mssdOnVideo(loadedModel,videoStream,streamlitOutput,streamlitFPSCounter,objectClasses,requiredConfidence) -> Optional[bool]:
    ''' 
    wrapper function for **MSSD** detection, executing modelOnVideo with 
    predefined function for mssd object detection
    '''
    modelOnVideo(wrapperRunningDnn,loadedModel,videoStream,streamlitOutput,streamlitFPSCounter,objectClasses,requiredConfidence)

# --- / 
# -- / 
# TODO add description accordingly 
def runYoloOnImage(loadedModel:object,requiredConfidence:float,objectClasses:list,imgObj,) -> dict:
    ''' 
    function running a selected model to detect a set of objects on a given image with a required confidence.
    Parameters include: 
    - loadedModel : yolo model 
    - objectClasses : list of all classes to search for 
    - imgObj : object of image to run on 
    - requiredConfidence : float indicating confidence threshold 
    
    **returns**: 
    - dictionary containg two keys:
    -> "image" -> resulted image 
    -> "foundObjects" -> all bounding boxes found
    '''
    
    # running model on given image 
    result:list= loadedModel.predict(imgObj,conf=requiredConfidence,classes=objectClasses)
    
    # gathering all found objects
    # TODO could be an massive slow down when run with VideoFeed! 
    # consider extracting this then? 
    resultedBoxes:list = result[0].boxes
    foundObjects:list = []
    for box in resultedBoxes:
        foundObjects.append(box.data)
    
    resultPlotted = result[0].plot()
    resultColorCorrected = cv2.cvtColor(resultPlotted, cv2.COLOR_BGR2RGB)
    
    # found objects is a dict with each class
    returnBlob: dict = {
        "image": resultColorCorrected,
        "foundObjects": foundObjects,
    }
    return returnBlob

# --- /
# -- / 
# TODO add description 
# TODO add function signature
def trainModel(model) -> Optional[str]:
    # TODO remove boolean-Blindness --> String values ambigous! 
    # TODO remove Stringvalue
    configFile:Optional[str] = gatherFilePath("**/yolov8_config.yaml")
    if configFile != None:
        # TODO adapt output path to save results in "preTrainedYolo"
        try: 
            savedPath = gatherFolderPath("**/preTrainedYolo")
            resultName:str = 'yolov8n_v8_50e'
            savingLocation = createPath(savedPath,result_name)
            results = model.train(
                data=gatherFilePath('**/yolov8_config.yaml'),
                imgsz=1280,
                epochs=1,
                batch=8,
                name= resultName,
                save_dir= savingLocation
                )
            
        except : 
            return "error running model"
        
    else: 
        # not path was found: 
        possibleError:Optional[str] = createYaml()
        if possibleError != None: 
            return possibleError
        # if created with no error, run again 
        return "Yaml was created, re run test once more"
    # return results


# --- /
# -- / 
# TODO --> refactor to another file 
# maybe smth like uiOfflineData? --> maybe its refactored too much at the end? 
# I have to concentrate on that a little later tho 
def offlineData(loadedModel):
    st.subheader("Offline Data Loading")
    st.write("This option will train the YOLO model using offline data.")
    st.write("Please make sure to provide the required data.yaml file.")
    st.write("Click the 'Train Model' button to start training.")

    if st.button("Train Model"):
        possibleError:Optional[str] = trainModel(loadedModel)
        if possibleError != None: 
            st.error(possibleError)
            return # aborting function
        # TODO improve verbosity to show more information ( what was trained, progress ..)
        st.write("Training complete!")



if __name__ == "__main__":
    exit("not meant to be run")