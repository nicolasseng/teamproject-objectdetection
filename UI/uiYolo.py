''' 
this file **temporarily** contains all the logic to provide a website build upon yolo v8. 

It will be removed once we have refactored and **united** our webinterface so that it can be run by all! 
''' 

# --- / 
# -- / external imports 
import streamlit as st
import glob

# --- / 
# -- / internal imports 
from modules.moduleFileManagement import gatherFilePath
from modules.moduleYoloV8 import initializeModel, runYoloOnImage
from UI.uiRunVideo import interfaceVideo

def runYoloInterface():
    # global variables
    # TODO get rid of global variables!
    confidence = 0.25
    # global model, classes

    st.title("Object Recognition Dashboard")
    st.sidebar.title("Settings")
    # confidence slider
    confidence = st.sidebar.slider(
        'Confidence', min_value=0.1, max_value=1.0, value=.45)
    
    
    try:
        defaultModelPath:str = gatherFilePath("**/yolov8s.pt")
    except:
        # path was not found 
        defaultModelPath:str = "yolov8s.pt"
    print(defaultModelPath)
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

    model_options = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "Custom"]

    yolo_model_selection:str = st.sidebar.selectbox("Select Yolov8 Model", model_options)

    if yolo_model_selection == "Custom":
        yolo_model:str = gatherFilePath('**/best.pt') 
    else:
        yolo_model:str = f"yolov8{yolo_model_selection.lower()[6:]}.pt"
        
    queriedPath = gatherFilePath( "**/"+yolo_model) # gathering path for selected yolo model 
    if queriedPath == None:
        # TODO construct path to download model to 
        # TODO refactor this path to settings! 
        initializeModel(yolo_model)
        st.error(f'Go to Offline Data to train your own data set or unzip the "runs" folder')
        return

    model = initializeModel(queriedPath)

    st.sidebar.markdown("---")

    ## --- ----
    ## ---- EXECUTING WITH SELECTED SOURCE 
    ## --- ----
    
    # input options
    # TODO refactor to check against list values instead of strings
    input_option = st.sidebar.radio(
        "Select input type: ", ['image', 'video', 'webcam', "YouTube Video", "Offline Data"])

    # input src option
    if input_option == "image" or input_option == "video":
        data_src = st.sidebar.radio("Select input source: ", [
                                    'Sample data', 'Upload your own data'])

    if input_option == 'image':
        img_file = None
        if data_src == 'Sample data':   
            # get all sample images
            img_path = glob.glob('data/sample_img/*')
            img_slider = st.slider("Select a test image.",
                                min_value=1, max_value=len(img_path), step=1)
            img_file = img_path[img_slider - 1]
        else:
            img_bytes = st.sidebar.file_uploader(
                "Upload an image", type=['png', 'jpeg', 'jpg'])
            if img_bytes:
                img_file = PIL.Image.open(img_bytes)

    # once image file was loaded or not 
        runYoloOnImage(model,classes,data_src,confidence)
        image_input(data_src,confidence)
    elif input_option == 'video':
        video_input(data_src,confidence)
    elif input_option == 'webcam':
         webcam()
    # elif input_option == 'YouTube Video':
        # youtube()
    elif input_option == "Offline Data":
        offlineData()
