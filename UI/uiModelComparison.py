'''
file implementing the UI of model performance comparison of an image
'''

# --- /
# -- / external imports
import streamlit as st
import PIL


def compareModels(confidenceThreshold:float):
    '''
    provides the UI for uploading an image, choosing a model 
    and starting the detection at the press of an button. 
    '''
    # TODO may become redundant when refactoring is finished
    imgFile = None
    imgBytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if imgBytes:
        imgFile = PIL.Image.open(imgBytes)
        st.image(imgFile, caption="Image")

    modelOptions = ["YOLOv8n", "YOLOv8s",
                    "YOLOv8m", "YOLOv8l", "Custom", "SSD"]
    modelOne = st.selectbox("Select first model", modelOptions)
    modelTwo = st.selectbox("Select second model", modelOptions)

    if st.button("Start Comparison"):
        if imgFile:
            metaData = detectObjects(imgFile, modelOne, modelTwo, confidenceThreshold)
            compareObjects(metaData)
        else:
            st.error("No image provided")


# TODO not sure if the types are correct
def detectObjects(image:PIL.Image, modelOne:str, modelTwo:str, confidenceThreshold:float) -> str:
    '''
    starts object detection using two models on the same image
    returns metadata such as detected objetcs and confidence values 
    '''
    col1, col2 = st.columns(2)
    with col1:
        # TODO: image is placeholder, implement prediction
        st.image(
            image, caption=f"[1]: Detection using {modelOne}", use_column_width=True)
    with col2:
        # TODO: image is placeholder, implement prediction
        st.image(
            image, caption=f"[2]: Detection using {modelTwo}", use_column_width=True)
    metadata = "TODO"  # TODO: placeholder: implement this
    return metadata


def compareObjects(metaData: str):
    '''
    compares metadata of the different models
    '''
    # TODO metadata should probably be a tupel or something similar + not sure what this should return
    st.write(f"Detected objects: {metaData}")
    # TODO: Implement object detection logic and display the results
    