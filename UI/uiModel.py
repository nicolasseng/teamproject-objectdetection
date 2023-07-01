''' 
this file **temporarily** contains all the logic to provide a website build upon yolo v8. 

It will be removed once we have refactored and **united** our webinterface so that it can be run by all! 
''' 

# --- / 
# -- / external imports 
from typing import Optional
from PIL import Image
import streamlit as st
import tempfile

from sympy import Union
from UI.uiRunningApp import displayMainWindow, displayMainWindowVideo

# --- / 
# -- / internal imports 
from modules.moduleFileManagement import convertArrayToImage, convertImageTo1DArray, gatherFilePath, gatherFolderContent, loadFromFile, prepareImageToSave, saveEvaluationToFile, saveToFile, convertListToArray
from modules.moduleYoloV8 import initializeModel, runYoloOnImage, offlineData
from UI.uiRunVideo import interfaceVideo

# --- /
# -- / 
# TODO REFACTOR TO uiRunning AP 
# TODO unite with MobileNetSSD Ui! 
# TODO add function signature 
# TODO add description 
# TODO refactor into smaller functions 
def runModelInterface():
    st.title("Object Recognition Dashboard")
    st.sidebar.title("Settings")
    
    # confidence slider

    st.sidebar.markdown("---")
     
    # input options
    SourceTypes:list = ['image', 'video', 'webcam', "YouTube Video", "Offline Data"]
    sourceTypeSelected,sourcePath  = selectSource(SourceTypes)
    st.sidebar.markdown("---")
    
    # gathering model to use 
    modelOptions:list = ["- Select model -","YoloV8","MobileNetSSD"]
    modelSelected: Optional[str] = st.sidebar.selectbox("Select Model to run", modelOptions)
    
    st.sidebar.markdown("---") 
     
    # default, waiting for input
    if modelSelected == modelOptions[0]: 
        st.markdown("# select model to start")
        return
    
    # gathering yolo model 
    if modelSelected == modelOptions[1]: 
        runOnYolo(SourceTypes,sourceTypeSelected,sourcePath)
        # selectionYoloModel()
    
    # gathering MSSD model
    if modelSelected == modelOptions[2]:
        runOnMobileNet(SourceTypes,sourceTypeSelected,sourcePath)

# --- / 
# -- / 
def runOnMobileNet(SourceTypes:list,selectedType:str,sourcePath:str|None = None) -> None:
    ''' 
    function being run, whenever mobilenetssd was selected as active model 
    this functino collects required data and then runs the model on the selected source
    '''
    confidence:float = gatherConfidence()
    if selectedType == SourceTypes[0]:
        # image input
        displayMainWindow(sourcePath,confidence)
    elif selectedType == SourceTypes[1]:
        # video input
        st.error("not done, sorry")
        # displayMainWindowVideo()
    
    elif selectedType ==  SourceTypes[2]:
        # webcam
        displayMainWindowVideo(confidence)
        
    elif selectedType == SourceTypes[3]:
        st.error("youtube loading was not implemented yet")
        return
    
    elif selectedType == SourceTypes[4]:
        st.error("Offline training has not been implemented for MobileNetSSD yet.")
        return None
    
 
# --- / 
# -- / 
def runOnYolo(SourceTypes:list,selectedType:str,sourcePath:str|None = None) -> None:
    '''
    function being run, whenever yolo was selected as active model
    this function collects the necessary data with user-inputs and then runs the selected source through the ObjectDetection 
    '''
    confidence:float = gatherConfidence()
    selectedModel = selectionYoloModel()
    if selectedModel == None: 
        return
    model:object =  selectedModel[1]
    modelName:str = selectedModel[0]
    gatheredClasses:list = gatherClassesYolo(model)
    
    if selectedType == SourceTypes[0]:
        
        processImage(model,gatheredClasses,sourcePath,confidence,modelName)
    
    elif selectedType == SourceTypes[1]:
        video_input(model,gatheredClasses,sourcePath,confidence)
    
    elif selectedType ==  SourceTypes[2]:
        processWebcam(model,gatheredClasses,confidence)
    
    elif selectedType == SourceTypes[3]:
        st.error("youtube loading was not implemented yet")
        return
    
    elif selectedType == SourceTypes[4]:
        # return 
        offlineData(model)
    
# --- / 
# -- / 
def runYoloOnObject(loadedModel:object,selectedClasses:list,source,confidence,modelName ):
    pass

def gatherClassesYolo(loadedModel:object) ->list :
    # -- /
    # gathering classes
    if st.sidebar.checkbox("Custom Classes"):
        model_names = list(loadedModel.names.values())
        assigned_class = st.sidebar.multiselect(
            "Select Classes", model_names, default=[model_names[0]])
        classes:list = [model_names.index(name) for name in assigned_class]
    else:
        classes:list = list(loadedModel.names.keys())
    
    return classes 

# --- / 
# -- / 
def selectionYoloModel() -> tuple[str,object] | None : 
    ''' 
    function displaying selection for choosing a YoloV8 model to run on 
    retursn the selected model initialized as Object 
    '''
    
    yoloOptions = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "Custom"]
    # defaults to Yolov8n
    yoloSelected: str = st.sidebar.selectbox("Select Yolov8 Model", yoloOptions,index=0)


    if yoloSelected != yoloOptions[4]: 
        yoloSelected = "{}.pt".format(yoloSelected.lower())
        pathToModel:str = gatherFilePath( "**/"+yoloSelected)
        
        if pathToModel == None:  # was not found, downloading accordingly
            st.error("File {} was not found on local storage, downloading: rerun afterwards".format(yoloSelected))
            initializeModel(yoloSelected)
            return 
            
    else:
        # custom trained set 
        pathToModel:Optional[str] = gatherFilePath('**/best.pt') 
        if pathToModel == None: 
            st.error(f'Go to Offline Data to train your own data set or unzip the "runs" folder')
            return  
            
    # case that path was found and set
    model = initializeModel(pathToModel)
    return (yoloSelected,model)


def selectSource(SourceTypes:list) -> tuple[str,str|None]: 
    '''
    function displaying selection for sourcetype and source 
    returning the selection as tuple 
    
    whenever the selected Source is **neither** an **image** or **video** the second value **will be none** 
    
    ### example usage: 
    selectSource() -> ("image", "SampleData")
    '''
    # setting default, in case we are not selecting video / image input
    selectedSource:str|None =None
    
    sourceTypeSelected: Optional[str] = st.sidebar.radio(
        "Select input type: ",SourceTypes,index=0 )
    
    sourceOptions:list =  ['Sample data', 'Upload your own data']
    
    if sourceTypeSelected == SourceTypes[0] or  sourceTypeSelected == SourceTypes[1] :
        
        sourceOptionSelected:str | None = st.sidebar.radio("Select input source: ",sourceOptions,index=0 )
    
    if sourceTypeSelected == SourceTypes[0]:
        selectedSource = str(selectFileSource(True,sourceOptions,sourceOptionSelected))
    
    if sourceTypeSelected == SourceTypes[1]: 
        selectedSource = str(selectFileSource(False,sourceOptions,sourceOptionSelected))

    
    return (sourceTypeSelected,selectedSource)
    
# --- / 
# -- / 
def gatherConfidence() -> float: 
    ''' 
    function displaying a slider to select confidence between 0.1 to 1 
    returns selected value as float
    '''
    confidence = st.sidebar.slider(
    'Confidence', min_value=0.1, max_value=1.0, value=.45)
    return confidence


# --- / 
# -- / 
def selectFileSource(isImage:bool,SourceOptions:list,selectedSource:Optional[str])-> str | object | None:
    '''
    function that queries file to use for detection. 
    It returns either a sample File (image or video) or an uploaded file (by the user )
    
    ### example usage: 
    selectFileSource(True,["sample img","ownData"],"sample img") -> will return a sample image depending on slider value
    '''
    # TODO get rid of sentinel values
    # TODO Refactor
    # initialize with sample image, preventing NONETYPE error
    defaultFile = gatherFilePath("**/sampleImg.jpg")
    if not isImage:
        defaultFile = gatherFilePath("**/sampleVid1.mp4")
    
                
    if selectedSource == SourceOptions[0]: # sample files
        # get all sample images
        queriedPath:str = "sample_img"
        if not isImage:
            queriedPath = "sample_vid"
        
        sampleFilesPaths:Optional[list] = gatherFolderContent(queriedPath)
        if sampleFilesPaths ==None:
            st.error("no sample files were found") 
            return None
        
        img_slider = st.slider("Select source.",
                                min_value=1, 
                                max_value=len(sampleFilesPaths),
                                step=1)
        # taking selected image 
        
        selectedFile = sampleFilesPaths[img_slider - 1]  
        return selectedFile
        # once image file was loaded or not 
    
    if selectedSource == SourceOptions[1]: # own upload 
        st.spinner("Waiting for your upload...")
        
        # default values for image 
        allowedFileType:list = ['png', 'jpeg', 'jpg']
        uploaderMessage:str = "Upload an image"
        
        if not isImage:
            allowedFileType:list = ['mp4', 'mpv', 'avi']
            uploaderMessage:str = "Upload a video"
        
        selectedFile = st.sidebar.file_uploader(
                uploaderMessage, type=allowedFileType)
        
        if selectedFile == None: 
            # setting default value
            return defaultFile
                
    # adapting file according to required filetype 
        if not isImage: 
            # creating temporary file for supplied video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(selectedFile.read())
            selectedFile = tfile.name
        else: 
            # creating image Object
            selectedFile = Image.open(selectedFile)
    

    
    return selectedFile
    
# --- /
# -- /
def processImage(loadedModel:object,objectClasses:list,selectedImage,confidence:float,usedModel:str):
     
    # once image file was loaded or not 
        col1, col2 = st.columns(2)
        with col1:
            st.image(selectedImage, caption="Selected Image", use_column_width=True)
        with col2:
            
            detectionResult = runYoloOnImage(loadedModel,objectClasses,selectedImage,confidence)
            st.image(detectionResult['image'], caption="Detected Image",
                     use_column_width=True)
            
            # --- 
            # - logic for evaluating results
            dictOfEvaluation:Optional[dict] = formEvaluateResult()
            if dictOfEvaluation == None: 
                return 
            
            # collecting all results: 
            
            # convertedImage:numpy.ndarray = convertImageTo1DArray(detectionResult["image"])
            # imageStringRepresentation:str = convertedImage.tolist()
            imageListRepresentation:list = prepareImageToSave(detectionResult["image"])
            imageDimension:tuple = detectionResult["image"].shape
            
            dictDetectionEval:dict= {
                "usedModel": usedModel,
                "confidence": confidence,
                "amountDetected": len(detectionResult["foundObjects"]),
                "actualAmount": dictOfEvaluation["amount"],
                "faultyDetection": dictOfEvaluation["faulty"],
                "imageArray": imageListRepresentation,
                "imageDimension": imageDimension
                }
            saveEvaluationToFile(dictDetectionEval,dictDetectionEval["usedModel"])
            
            # --- / 
            # -- / testing only!!
            # pathToTest = gatherFilePath("**/testEvaluation.json")
            # maybeDict:Optional[dict] = loadFromFile(pathToTest)
            
            # extractedArray = convertListToArray(maybeDict["imageArray"])
            # arrayDimension = maybeDict["imageDimension"]
            # imageArray = convertArrayToImage(extractedArray,arrayDimension)
            # st.image(imageArray,"extracted image!!")

# --- / 
# -- / 
# TODO refactor to another file --> does not belong here! 
# TODO add function description 
# TODO add signature 
def video_input(loadedModel:object,objectClasses:list,selectedVideo,confidence:float):
        interfaceVideo(loadedModel,selectedVideo,objectClasses,confidence)

# --- / 
# -- / 
def processWebcam(loadedModel:object,objectClasses:list,requiredConfidence:float):
    ''' 
    function runs Detectionmodel on Webcam. 
    Opens the webcam and then pipes its datastream into the "interfaceVideo"
    '''
    interfaceVideo(loadedModel,None,objectClasses,requiredConfidence)
    

# dictionary returned should contain: 
# - usedModel -> to be displayed upon showcase
# - confidence level : 
# - detected amount of objects
# - actual amount of objects detected -> user input!
# - amount of faulty detection -> user input!
# - image encoded as 1D-array 
# - image dimensions used to reshape accordingly
# --- / 
# -- / 
def formEvaluateResult()-> Optional[dict]: 
    with st.form("evaluating results"):
        st.write("insert the correct data for this detected image")
        st.write("once done, click the submit-button to save the result")
        actualDetection:int = int(st.number_input("actual amount of detections",min_value=0,max_value=20,step=1))
        amountFaultyDetection:int = int(st.number_input("amount of faulty detections",min_value=0,max_value=20,step=1))
        submitted = st.form_submit_button("save results")
        if submitted:
            resultDict:dict = {
                "amount": actualDetection,
                "faulty": amountFaultyDetection
            }
            return resultDict
        return None