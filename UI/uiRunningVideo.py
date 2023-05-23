''' 
file contains logic for streamlit to display an according website 
'''
# --- /
# -- / internal imports 
import settings.modelSettings as msettings
import modules.moduleDetectionMobileNetSSD as MSSD
from settings.modelSettings import MSSDnetwork, MSSDWeight,gatherFilePath
# --- / 
# -- / external imports 
import streamlit as st
from io import StringIO
from PIL import Image
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

# --- /
# -- / 

def run_the_app():
    st.sidebar.markdown("# Settings")
    #TODO refactor further
    FileSelection:list =  ["sample image", "upload image","supply video"]
    imageSource = st.sidebar.radio(
        "Select file:",FileSelection
    )
    
    # --- /  styling sidebar
    
    input_option = st.sidebar.radio(
        "Select input type: ", ["image", "video", "livestream"]
    )
    #TODO selection of possible images!
    
    st.sidebar.markdown("# Model")
    # selecting specific classes to search for 
    #TODO implement into interface for object detection model later
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", min_value=0.1, max_value=1.0, value=0.45
        )
    
    model_names = list({"bikes", "car", "Traffic light", "human"})
    if st.sidebar.checkbox("Search for specific objects?"):
        assigned_class = st.sidebar.multiselect(
            "Select objects", model_names, default=[model_names[0]]
        )
        classes = [model_names.index(name) for name in assigned_class]
    
    if imageSource == FileSelection[0]:
        st.sidebar.markdown("### available demo images")
        #TODO add carousel of images available
        
        # select default image to display:
        imagePath = gatherFilePath("**/sampleImg*.jpg")
        displayMainWindow(imagePath,confidence_threshold)
    
    if imageSource == FileSelection[1]:
        #style if Sources are uploads
        st.sidebar.markdown("### upload your image")
        # setting uploaded image as the one to display
        imageUploaded = uploadImage()
        # waiting until file was uploaded 
        print("file was uploaded")
        displayMainWindow(imageUploaded,confidence_threshold)
    if imageSource == FileSelection[1]:
        st.sidebar.markdown("###supplying video")
        # setting uploaded image as the one to display
        # waiting until file was uploaded 
        print("video activated")
        displayMainWindow(confidence_threshold)
        
    # defining Main windows
    
    # gathering file to display:
    #TODO add depency for selected image from carousel 
    # imageDisplayed = gatherFilePath("**/sampleImg*.jpg")
    
    # TODO insert model to process here // 
    
    

def displayMainWindow(confidence_threshold:float):
    
    # loading model 
    
    loadedNet = MSSD.loadModel(MSSDnetwork,MSSDWeight)
    MSSD.runOnVideoStream(loadedNet,confidence_threshold)
    
    
    # displaying results 
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            imageUnprocessed,
            use_column_width="auto",
            caption="Image without object detection",
        )
    with col2:
        st.image(
            imageProcessed,
            use_column_width="auto",
            caption="image with object detection"
        )
    


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
