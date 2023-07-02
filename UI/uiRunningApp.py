''' 
file contains logic for streamlit to display an according website 
'''
from io import StringIO
from typing import Optional

# for video input --> not the optimal solution
import cv2 as opencv
import numpy as np
import pandas as pd
# --- / 
# -- / external imports 
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

# --- /
# -- / internal imports 
import modules.moduleDetectionMobileNetSSD as MSSD
import settings.modelSettings as msettings
from modules.moduleFileManagement import gatherFolderContent
from settings.modelSettings import MSSDnetwork, MSSDWeight, gatherFilePath
from .uiModelComparison import compareModels

# --- /
# -- / 

def run_the_app():
    st.sidebar.markdown("# Settings")
    #TODO refactor further

    confidence_threshold = .45
    inputOptionSelected: Optional[str] = None
    inputTypes:list =  ["image", "webcam"]
    inputTypeSelected = st.sidebar.radio(
        "Select input type:", inputTypes, key="source"
    )
    
    # --- /  styling sidebar
    # --- / 
    # -- / formatting Main-window accordingly 
    if inputTypeSelected == inputTypes[0]:
        sourceOptions: list = ["Sample data", "Upload your own data"]
        if inputTypeSelected == inputTypes[0] or  inputTypeSelected == inputTypes[1] :
            inputOptionSelected = st.sidebar.radio("Select input source: ", sourceOptions, key="upload")

        if inputOptionSelected == sourceOptions[0]: # sample images
            #TODO add carousel of images available
            
            # select default image to display:
            imagePath: list = gatherFolderContent("sample_img")
            count = len(imagePath)
            image: list = list()
            for i in range(count):
                image.append("Image " + str(i+1))
            images = st.sidebar.radio("Select sample image", image)
            file = imagePath[int(images[-1])-1]
            displayMainWindow(file,confidence_threshold)
        
        if inputOptionSelected == sourceOptions[1]: # uploading images
            #style if Sources are uploads
            st.sidebar.markdown("### upload your image")
            # setting uploaded image as the one to display
            imageUploaded = uploadImage()
            # waiting until file was uploaded 
            print("file was uploaded")
            displayMainWindow(imageUploaded,confidence_threshold)       
    elif inputTypeSelected == inputTypes[1]: # supplying video stream
        st.sidebar.markdown("### starting and querying webcam")
        print("running webcams")
        displayMainWindowVideo(confidence_threshold)
        
    st.sidebar.markdown("# Model")
    # selecting specific classes to search for 
    #TODO implement into interface for object detection model later
    confidence_threshold = st.sidebar.slider(
        "Confidence slider", min_value=0.1, max_value=1.0, value=0.45, key="conf"
        )
    

    model_names = list({"bikes", "car", "Traffic light", "human"})
    if st.sidebar.checkbox("Search for specific objects?"):
        assigned_class = st.sidebar.multiselect(
            "Select objects", model_names, default=[model_names[0]]
        )
        classes = [model_names.index(name) for name in assigned_class]
    
    
def displayMainWindowVideo(confidence_threshold:float):
    column1,column2 = st.columns(2)
    loadedNet = MSSD.loadModel(MSSDnetwork,MSSDWeight)
    # setting video input
    VideoStream = opencv.VideoCapture(0)
    result, initialFrame = VideoStream.read()
    # empty image container to fill during runtime
    displayedImage = st.image(initialFrame)
    
    
    with column1:
        
        displayedImage
        # running model which will update the piped imageobject
        MSSD.wrapperRunningDnn(loadedNet,confidence_threshold,videoStream=VideoStream,streamlitOutput=displayedImage)

        
        
        
    

def displayMainWindow(imageUploaded,confidence_threshold:float):
    if imageUploaded == None:
        # no file was uploaded yet 
        imageUploaded = gatherFilePath("**/sample*.jpg")
    # preparing supplied image
    imageUploaded = Image.open(imageUploaded) # opening image to convert 
     # creating two identical copies, showing un/processed images at the end 
    imageProcessed = np.array(imageUploaded)
    imageUnprocessed = np.array(imageUploaded)
    
    col1, col2 = st.columns(2)
    with col1:
        # defining Images to set 
        stUnprocessImage = st.image(
                imageUnprocessed,
                use_column_width="auto",
                caption="Image without object detection",
            )
    
    with col2:
        # setting default value first 
        stProcessedImage = st.image(
                imageUnprocessed,
            )
    
    # TODO get model loading out from webinterface --> create wrapper function which chooses
    # according to given model as string! --> helps implement Yolo and further models into this structure 
    loadedNet = MSSD.loadModel(MSSDnetwork,MSSDWeight)
    # running model which will update the piped imageobject
    MSSD.wrapperRunningDnn(loadedNet,confidence_threshold,imageObj=imageProcessed,streamlitOutput=stProcessedImage)


def uploadImage():
    imageFile = st.file_uploader("Upload an Image",type=["jpg","png","jpeg","avi"]) 
    # check for input
    if not imageFile:
        return None 
    # if image was supplied: 
    return imageFile

# --- / 
# -- /
def upload_file_sidebar():
    uploaded_file = st.sidebar.file_uploader("Choose a file:")
    if uploaded_file is not None:
        # To read file as bytes:
        data = uploaded_file.getvalue()
        st.write(data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        st.info("File upload was successfull")
