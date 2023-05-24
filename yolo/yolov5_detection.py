import torch
import numpy as np
import cv2
import time
import streamlit as st


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels, Coordinates, and Confidence scores of objects detected by the model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord, scores = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.xyxyn[0][:, 4]
        return labels, cord, scores

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes, labels, and confidence scores on the frame.
        :param results: contains labels, coordinates, and confidence scores predicted by the model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes, labels, and confidence scores plotted on it.
        """
        labels, cord, scores = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                label = self.class_to_label(labels[i])
                confidence = scores[i]
                text = f"{label}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def detect_video(self, video_path):
        """
        Reads frames from a video file and detects objects in each frame.
        :param video_path: Path to the video file.
        :return: None
        """
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_webcam(self):
        """
        Reads frames from the webcam and detects objects in each frame.
        :return: None
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def __call__(self, input_source):
        """
        This function is called when the class is executed. It runs the loop to read frames from the specified input
        source and detect objects in each frame.
        :param input_source: 'video' for video file or 'webcam' for webcam input.
        :return: void
        """
        if input_source == 'video':
            video_path = 'test.mp4'  # Replace with the actual path to your video file.
            self.detect_video(video_path)
        elif input_source == 'webcam':
            self.detect_webcam()
        else:
            print("Invalid input source!")


# Create a new object and execute.
detection = ObjectDetection()

def run_yolo():
    st.title("Object Detection with YOLOv5")
    st.header("Press q to close pop up window")
    input_source = st.sidebar.radio("Select Input Source", ("Video", "Webcam"))

    if input_source == "Video":
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            video_path = f"uploaded_video{uploaded_file.name[-4:]}"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.video(video_path)

    if st.button("Start Detection"):
        st.write("Running object detection...")
        with st.spinner("Loading model..."):
            # Load the model (cached)
            @st.cache_data
            def load_model():
                return detection.load_model()

            model = load_model()

        if input_source == "Video" and uploaded_file is not None:
            with st.spinner("Processing video..."):
                detection.detect_video(video_path)
        else:
            with st.spinner("Processing webcam..."):
                detection.detect_webcam()

        st.success("Object detection completed.")