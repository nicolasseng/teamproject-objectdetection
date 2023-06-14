"""
file contains functions and structures necessary for running yolov8 or similar versions
with our object detection webapp. More specifically it is only supplying the logic 
necesssary and is later piped or accessed by the UI.  
"""

# --- /
# -- / external imports 
from typing import Optional
import cv2 
import time
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os
# TODO to be removed once refactored! 
import streamlit as st
# TODO remove those libraries because they are not needed here --> at least should not be 
import PIL 
import glob 
import tempfile

# --- / 
# -- / internal imports 
from modules.moduleFileManagement import gatherFilePath
from modules.moduleYamlManagement import createYaml

# --- / 
# -- / 

#TODO DEPRECATED ?? 
def load_model(yolo_model:str):
    '''
    takes string to trained yolo detection model 
    returns loaded model for further use 
    
    ### sample usage  : 
    load_model("path/to/yolov8n.pt) -> returns accordingly loaded yolo detection model
    '''
    model = YOLO(yolo_model)
    return model

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
    print(selectedModelPath)
    # model:object = load_model(selectedModelPath)
    model:object = YOLO(selectedModelPath)
    return model
    



# TODO DEPRECATED
def infer_image(loadedModel,confidence:float,frame,objectClass,requiredConfidence, output):
    frame = cv2.resize(frame, (720, int(720*(9/16))))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # added yoloRun which initializes and runs the model on the given frame 
    result:dict = runYoloOnImage(loadedModel,objectClass,frame,requiredConfidence)
    # res = loadedModel.predict(frame, conf=confidence, classes=classes)
    # res_plotted = res[0].plot()
    
    # updating output according to returned result
    output.image(result['image'], caption="Detection in Video", use_column_width=True)


def processFrame(image): 
    '''
    process Image such as it is required by videostreams  
    returns processed image
    '''
    image = cv2.resize(frame, (720, int(720*(9/16))))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return image



# --- / 
# -- / 
# TODO refactor only contain logic, nothing to display anymore! 
# TODO split running yolo and website creation 
# --> something like adding entry to run_yolov8() in uiRunningApp.py 
# which is then gathering all information etc
# TODO add function signature 
# TODO add description 
# TODO refactor into smaller functions 
# TODO 
def run_yolov8():
    # global variables
    # TODO get rid of global variables!
    confidence = 0.25
    global model, classes

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

    # custom classes
    if st.sidebar.checkbox("Custom Classes"):
        model_names = list(model.names.values())
        assigned_class = st.sidebar.multiselect(
            "Select Classes", model_names, default=[model_names[0]])
        classes = [model_names.index(name) for name in assigned_class]
    else:
        classes = list(model.names.keys())

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
        image_input(data_src,confidence)
    elif input_option == 'video':
        video_input(data_src,confidence)
    elif input_option == 'webcam':
         webcam()
    # elif input_option == 'YouTube Video':
        # youtube()
    elif input_option == "Offline Data":
        offlineData()

# --- / 
# -- / 
def yoloOnVideo(loadedModel,videoStream,streamlitOutput,streamlitFPSCounter,objectClasses,requiredConfidence) -> Optional[bool]:
    '''
    this function takes a videostream and processes its frames by running them on a loaded yoloV8 model 
    this videoStream can either be a **video** or **webcam input** 
    once it is done processing it will return a bool indicating that it stopped!  
    '''
    # reading from given videoStream 
    timeBeforeProcessing:float = time.time()
    while videoStream.isOpened():
        
        result, frame = videoStream.read() 
        
        if not result: # not parsing if nothing was received 
            return None # --> indicates issue!
        # processing frame received 
        processedFrame= processFrame(frame)
        # running detection on it 
        result = runYoloOnImage(loadedModel,objectClasses,processedFrame,requiredConfidence)
        # infer_image(loadedModel,requiredConfidence,frame,streamlitOutput)
        
        # displaying resulted detection with StreamlitOutput!
        streamlitOutput.image(result['image'], caption="Detection in Video", use_column_width=True)
        
        timeAfterProcessing:float = time.time()
        #calculate fps 
        currentFPS:float = round(1/ (timeAfterProcessing - timeBeforeProcessing),2)
        #display fps 
        streamlitFPSCounter.markdown(("**{}**".format(currentFPS)))
    
    return True


# --- / 
# -- / 
# TODO refactor to another file --> does not belong here! 
# TODO add function description 
# TODO add signature 
def image_input(data_src,confidence:float):
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
    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image", use_column_width=True)
        with col2:
            res = model.predict(img_file, conf=confidence, classes=classes)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption="Detected Image",
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No image is uploaded yet!")


# --- / 
# -- / 
def runYoloOnImage(loadedModel:object,objectClasses:list,imgObj,requiredConfidence:float=0.4,) -> dict:
    # running model on given image 
    result:list= loadedModel.predict(imgObj,conf=requiredConfidence,classes=objectClasses)
    
    # gathering all found objects
    resultedBoxes:list = result[0].boxes
    foundObjects:list = []
    for box in resultedBoxes:
        foundObjects.append(box.data)
    
    resultPlotted = result[0].plot()
    
    # found objects is a dict with each class and its color
    returnBlob: dict = {
        "image": resultPlotted,
        "foundObjects": foundObjects,
    }
    return returnBlob

# --- / 
# -- / 
# TODO refactor to another file --> does not belong here! 
# TODO add function description 
# TODO add signature 
def video_input(data_src,confidence:float):
    vid_file = None
    tfile = None
    # selecting data to stream 
    # TODO refactor to webinterface! 
    
    if data_src == 'Sample data':
        vid_file = "data/sample_vid/sample.mp4"
    else:
        st.spinner("Waiting for your upload...")
        vid_bytes = st.sidebar.file_uploader(
            "Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_bytes.read())
            vid_file = tfile.name

    # processing supplied video 
    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # fps calculation 
        fps = 0
        prev_time = 0
        curr_time = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")
        st.markdown("---")
        
        output = st.empty()
        
        while cap.isOpened(): # exactly the same procedure as with 
            ret, frame = cap.read()
            if ret:
                infer_image(frame, output)
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                st1_text.markdown(f"**{height}**")
                st2_text.markdown(f"**{width}**")
                st3_text.markdown(f"**{fps:.2f}**")
            else:
                st.write("Can't read frame, stream ended? Exiting ....")
                break

        cap.release()




# --- /
# -- / 
# TODO add description 
# TODO add function signature
def trainModel(model):
    # Load the model.
    model = YOLO('yolov8n.pt')
    if modules.moduleFileManagement.readFile('yolov8_config.yaml') == 'file not found':
        createYaml()

    # Training.
    results = model.train(
        data=modules.moduleFileManagement.gatherFilePath('**/yolov8_config.yaml'),
        imgsz=1280,
        epochs=1,
        batch=8,
        name='yolov8n_v8_50e'
        )

    return results


# --- /
# -- / 
# TODO --> refactor to another file 
# maybe smth like uiOfflineData? --> maybe its refactored too much at the end? 
# I have to concentrate on that a little later tho 
def offlineData():
    st.subheader("Offline Data Loading")
    st.write("This option will train the YOLO model using offline data.")
    st.write("Please make sure to provide the required data.yaml file.")
    st.write("Click the 'Train Model' button to start training.")

    if st.button("Train Model"):
        try:
            trainModel(model)
            st.write("Training complete!")
        except RuntimeError:
            st.error('No data set provided or the "data" folder is not unzipped')


if __name__ == "__main__":
    try:
        run_yolov8()
    except SystemExit:
        pass