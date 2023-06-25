''' 
this file **temporarily** contains all the logic to provide a website build upon yolo v8. 

It will be removed once we have refactored and **united** our webinterface so that it can be run by all! 
''' 

import tempfile
# --- / 
# -- / external imports 
from typing import Optional

import streamlit as st
from numpy import select  # ought to be removed at a later point
from PIL import Image

# --- / 
# -- / internal imports 
from modules.moduleFileManagement import gatherFilePath, gatherFolderContent
from modules.moduleYoloV8 import initializeModel, offlineData, runYoloOnImage
from UI.uiRunningApp import run_the_app
from UI.uiRunVideo import interfaceVideo


# --- /
# -- / 
# TODO REFACTOR TO uiRunning AP 
# TODO unite with MobileNetSSD Ui! 
# TODO add function signature 
# TODO add description 
# TODO refactor into smaller functions 
def runYoloInterface():
    # global variables
    # TODO get rid of global variables!
    confidence = 0.25
    # global model, classes

    st.title("Object Recognition Dashboard")
    st.sidebar.title("Settings")

    objectDetectionSelected: Optional[str] = None
    ObjectDetection:list = ['Yolo', 'SSD']
    sourceTypeSelected: Optional[str] = st.sidebar.radio(
        "Select Object Detection: ", ObjectDetection)
    
    if sourceTypeSelected == 'SSD':
        run_the_app()
        return
    
    # confidence slider
    confidence = st.sidebar.slider(
        'Confidence', min_value=0.1, max_value=1.0, value=.45)
    
    
    # TODO refactor
    # -- / 
    # loads model 
    try:
        defaultModelPath:str = gatherFilePath("**/yolov8s.pt")
    except:
        # path was not found 
        defaultModelPath:str = "yolov8s.pt"
    print(defaultModelPath)
    # TODO refactor
    # only load model after selection of one --> takes longer to load otherwise!
    model = initializeModel(defaultModelPath)

    # -- /
    # gathering classes
    if st.sidebar.checkbox("Custom Classes"):
        model_names = list(model.names.values())
        assigned_class = st.sidebar.multiselect(
            "Select Classes", model_names, default=[model_names[0]])
        classes:list = [model_names.index(name) for name in assigned_class]
    else:
        classes:list = list(model.names.keys())

    st.sidebar.markdown("---")
    
    # -- / 
    # gathering yolo model 
    modelOptions = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "Custom"]
    modelSelected:str = st.sidebar.selectbox("Select Yolov8 Model", modelOptions,index=0)


    if modelSelected != modelOptions[4]: 
        modelSelected = "{}.pt".format(modelSelected.lower())
        pathToModel:str = gatherFilePath( "**/"+modelSelected)
        
        if pathToModel == None:  # was not found, downloading accordingly
            st.error("File {} was not found on local storage, downloading: rerun afterwards".format(modelSelected))
            initializeModel(modelSelected)
            return 
            
        
    else:
        # custom trained set 
        pathToModel:Optional[str] = gatherFilePath('**/best.pt') 
        if pathToModel == None: 
            st.error(f'Go to Offline Data to train your own data set or unzip the "runs" folder')
            return  
            
    # case that path was found and set
    model = initializeModel(pathToModel)

    st.sidebar.markdown("---")

    ## --- ----
    ## ---- EXECUTING WITH SELECTED SOURCE 
    ## --- ----
    
    # input options
    # initial values
    sourceOptionSelected: Optional[str] = None
    # TODO refactor to check against list values instead of strings
    SourceTypes:list = ['image', 'video', 'webcam', "YouTube Video", "Offline Data"]
    sourceTypeSelected: Optional[str] = st.sidebar.radio(
        "Select input type: ",SourceTypes )

    # TODO refactor to separate function! 
    # TODO refactor to list selection
    # input src option
    sourceOptions:list =  ['Sample data', 'Upload your own data']
    if sourceTypeSelected == SourceTypes[0] or  sourceTypeSelected == SourceTypes[1] :
        sourceOptionSelected = st.sidebar.radio("Select input source: ",sourceOptions )

    
    # gathering image
    # TODO refactor to separate function!
    if sourceTypeSelected == SourceTypes[0]:
        selectedImage = selectFileSource(True,sourceOptions,sourceOptionSelected)
        processImage(model,classes,selectedImage,confidence)
    
    elif sourceTypeSelected == SourceTypes[1]:
        selectedVideo = selectFileSource(False,sourceOptions,sourceOptionSelected)
        video_input(model,classes,selectedVideo,confidence)
    
    elif sourceTypeSelected ==  SourceTypes[2]:
        processWebcam(model,classes,confidence)
    
    elif sourceTypeSelected == SourceTypes[3]:
        st.error("youtube loading was not implemented yet")
        return
    
    elif sourceTypeSelected == SourceTypes[4]:
        # return 
        offlineData(model)

# --- / 
# -- / 
def selectFileSource(isImage:bool,SourceOptions:list,selectedSource:Optional[str]):
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
        
        sampleFilesPaths:list = gatherFolderContent(queriedPath)
        if isImage:
            sampleImageList:list = list()
            for i in range(len(sampleFilesPaths)):
                sampleImageList.append("Image " + str(i+1))
            selection = st.sidebar.radio(
                "Select image.", sampleImageList)
        else: 
            sampleVideoList:list = list()
            for i in range(len(sampleFilesPaths)):
                sampleVideoList.append("Video " + str(i+1))
            selection = st.sidebar.radio(
                "Select video.", sampleVideoList)
        
        # taking selected image 
        
        selectedFile = sampleFilesPaths[int(selection[-1])-1]

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
    
    
    
def processImage(loadedModel:object,objectClasses:list,selectedImage,confidence:float):
     
    # once image file was loaded or not 
        col1, col2 = st.columns(2)
        with col1:
            st.image(selectedImage, caption="Selected Image", use_column_width=True)
        with col2:
            
            detectionResult = runYoloOnImage(loadedModel,objectClasses,selectedImage,confidence)
            st.image(detectionResult['image'], caption="Detected Image",
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in detectionResult['foundObjects']:
                        st.write(box)
            
            except Exception as ex:
                st.write("No image was uploaded yet!")


# --- / 
# -- / 
# TODO refactor to another file --> does not belong here! 
# TODO add function description 
# TODO add signature 
def video_input(loadedModel:object,objectClasses:list,selectedVideo,confidence:float):
        interfaceVideo(loadedModel,selectedVideo,objectClasses,confidence)
        # cap = cv2.VideoCapture(videoFile)
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # # fps calculation 
        # fps = 0
        # prev_time = 0
        # curr_time = 0
        # st1, st2, st3 = st.columns(3)
        # with st1:
        #     st.markdown("## Height")
        #     st1_text = st.markdown(f"{height}")
        # with st2:
        #     st.markdown("## Width")
        #     st2_text = st.markdown(f"{width}")
        # with st3:
        #     st.markdown("## FPS")
        #     st3_text = st.markdown(f"{fps}")
        # st.markdown("---")
        
        # output = st.empty()
        # yoloOnVideo()
        # while cap.isOpened(): # exactly the same procedure as with 
        #     ret, frame = cap.read()
        #     if ret:
        #         infer_image(frame, output)
        #         curr_time = time.time()
        #         fps = 1 / (curr_time - prev_time)
        #         prev_time = curr_time
        #         st1_text.markdown(f"**{height}**")
        #         st2_text.markdown(f"**{width}**")
        #         st3_text.markdown(f"**{fps:.2f}**")
        #     else:
        #         st.write("Can't read frame, stream ended? Exiting ....")
        #         break

        # cap.release()

# --- / 
# -- / 
def processWebcam(loadedModel:object,objectClasses:list,requiredConfidence:float):
    ''' 
    function runs Detectionmodel on Webcam. 
    Opens the webcam and then pipes its datastream into the "interfaceVideo"
    '''
    interfaceVideo(loadedModel,None,objectClasses,requiredConfidence)
    