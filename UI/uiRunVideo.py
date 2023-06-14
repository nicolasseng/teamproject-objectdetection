'''
file contains Ui foundation to run a webcame detection. 
It supplies the structure of our webapp in order to supply a model  
'''

# --- /
# -- / external imports 
import cv2 
import streamlit as st
import time

from modules.moduleYoloV8 import yoloOnVideo


# --- / 
# -- / internal imports 


# TODO requires annotation and function signature 
# TODO requires better / more descriptive name 
# TODO add description and example usage ? 

def interfaceVideo(loadedModel,videoStream,objectClasses,requiredConfidence):
    ''' 
    this function takes a loaded model to use --> could be any yoloV8 or similar
    further takes a video stream( either Webcam or supplied video!) and runs object detection with
    the given model on it.
    
    outputs the results with Streamlit
    '''
    # opening video stream <
    # videoStream = cv2.Videocapture(0) #
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if custom_size:
        width = st.sidebar.number_input(
            "Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input(
            "Height", min_value=120, step=20, value=height)
    
    # creating three columns 
    col0, col1, col2 = st.columns(3)
    with col0:
        st.markdown("## Height")
        col0_text = st.markdown(f"{height}")
    with col1:
        st.markdown("## Width")
        col1_text = st.markdown(f"{width}")
    with col2:
        st.markdown("## FPS")
        col2_text = st.markdown("0") # default is 0

    # setting divider
    st.markdown("---")
    
    # output image 
    output = st.empty()
    
    # loading model before use ! 
    
    terminatedVideoStream = yoloOnVideo(loadedModel,videoStream,output,col2_text,object,requiredConfidence)
    if terminatedVideoStream: 
        # closing videoStream
        videoStream.release()
