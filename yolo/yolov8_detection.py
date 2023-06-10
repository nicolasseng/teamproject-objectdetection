import glob
import tempfile
import time

import cv2
import pafy
import PIL
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os
import modules.moduleFileManagement
from .yamlCreation import createYaml

confidence = 0.25

# Load the YOLO model


def load_model(yolo_model):
    model = YOLO(yolo_model)
    return model


def infer_image(frame, output):
    frame = cv2.resize(frame, (720, int(720*(9/16))))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = model.predict(frame, conf=confidence, classes=classes)
    res_plotted = res[0].plot()
    output.image(res_plotted, caption="Detected Video", use_column_width=True)


def image_input(data_src):
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


def video_input(data_src):
    vid_file = None
    tfile = None
    if data_src == 'Sample data':
        vid_file = "data/sample_vid/sample.mp4"
    else:
        st.spinner("Waiting for your upload...")
        vid_bytes = st.sidebar.file_uploader(
            "Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_bytes.read())
            # vid_file = "data/sample_vid/upload." + vid_bytes.name.split('.')[-1]
            # with open(vid_file, 'wb') as out:
            #     out.write(vid_bytes.read())

    if vid_file or tfile:
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 0
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
        prev_time = 0
        curr_time = 0
        while cap.isOpened():
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


def webcam():
    cap = cv2.VideoCapture(0)
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if custom_size:
        width = st.sidebar.number_input(
            "Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input(
            "Height", min_value=120, step=20, value=height)

    fps = 0
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
    prev_time = 0
    curr_time = 0
    while cap.isOpened():
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


# Does not work yet
""" def youtube():
    source_youtube = st.sidebar.text_input("YouTube url")
    video = pafy.new(source_youtube)
    best = video.getbest(preftype="mp4")
    vid_cap = cv2.VideoCapture(best.url)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st_frame = st.empty()
    fps = 0
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
    while(vid_cap.isOpened()):
        ret, frame = vid_cap.read()
        if not ret:
            st.write("Can't read frame, stream ended? Exiting ....")
            break
        frame = cv2.resize(frame, (720, int(720*(9/16))))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = model.predict(frame, conf=confidence, classes=classes)
        res_plotted = res[0].plot()
        st_frame.image(res_plotted, caption="Detected Video", use_column_width=True)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")
    vid_cap.release() """


# Offline Data loading
def offline_data(model):
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


def run_yolov8():
    # global variables
    global model, confidence, classes

    st.title("Object Recognition Dashboard")

    st.sidebar.title("Settings")

    # confidence slider
    confidence = st.sidebar.slider(
        'Confidence', min_value=0.1, max_value=1.0, value=.45)

    model = load_model("yolov8s.pt")

    # custom classes
    if st.sidebar.checkbox("Custom Classes"):
        model_names = list(model.names.values())
        assigned_class = st.sidebar.multiselect(
            "Select Classes", model_names, default=[model_names[0]])
        classes = [model_names.index(name) for name in assigned_class]
    else:
        classes = list(model.names.keys())

    st.sidebar.markdown("---")

    yolo_model_selection = st.sidebar.radio(
        "Select Yolov8 Model", ("YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "Custom"))

    if yolo_model_selection == "YOLOv8n":
        yolo_model = "yolov8n.pt"
    elif yolo_model_selection == "YOLOv8s":
        yolo_model = "yolov8s.pt"
    elif yolo_model_selection == "YOLOv8m":
        yolo_model = "yolov8m.pt"
    elif yolo_model_selection == "YOLOv8l":
        yolo_model = "yolov8l.pt"
    elif yolo_model_selection == "Custom":
        yolo_model = "C:/Users/frede/Documents/new_team/runs/detect/yolov8n_v8_50e/weights/best.pt"  # your path

    model = load_model(yolo_model)

    st.sidebar.markdown("---")

    # input options
    input_option = st.sidebar.radio(
        "Select input type: ", ['image', 'video', 'webcam', "YouTube Video", "Offline Data"])

    # input src option
    if input_option == "image" or input_option == "video":
        data_src = st.sidebar.radio("Select input source: ", [
                                    'Sample data', 'Upload your own data'])

    if input_option == 'image':
        image_input(data_src)
    elif input_option == 'video':
        video_input(data_src)
    elif input_option == 'webcam':
        webcam()
    elif input_option == 'YouTube Video':
        youtube()
    elif input_option == "Offline Data":
        st.subheader("Offline Data Loading")
        st.write("This option will train the YOLO model using offline data.")
        st.write("Please make sure to provide the required data.yaml file.")
        st.write("Click the 'Train Model' button to start training.")

        if st.button("Train Model"):
            offline_data(model)
            st.write("Training complete!")


if __name__ == "__main__":
    try:
        run_yolov8()
    except SystemExit:
        pass
