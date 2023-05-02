import os
import urllib
from io import StringIO

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Object Detection Web App", layout="wide")

# HEADER
st.subheader("Object Detection Web App")

st.write("Open the Sidebar to either upload a File or show the source code.")


def main():
    # readme_text = st.markdown(get_file_content_as_string("README.md"))
    st.sidebar.title("What to do?")
    app_mode = st.sidebar.selectbox(
        "What do you want to do?", ["Run the App", "Show the Code", "Upload a File"]
    )
    if app_mode == "Run the App":
        run_the_app()
    elif app_mode == "Show the Code":
        # readme_text.empty()
        # st.code(get_file_content_as_string("test.py"))
        st.write("Working on it...")
    elif app_mode == "Upload a File":
        upload_file()


def upload_file():
    uploaded_file = st.file_uploader("Choose a file")
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


def get_file_content_as_string(path):
    url = (
        "https://raw.githubusercontent.com/nicolasseng/teamproject-objectdetection/main/"
        + path
    )
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def run_the_app():
    st.sidebar.markdown("# Frame")
    object_type = st.sidebar.selectbox(
        "Search for which objects?",
        {
            "biker": "biker",
            "car": "car",
            "trafficLight": "traffic light",
            "human": "human",
        },
        2,
    )
    min_elts, max_elts = st.sidebar.slider(
        "How many %ss (select a range)?" % object_type, 0, 25, [10, 20]
    )
    selected_frame_index = st.sidebar.slider("Choose a frame (index)")
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01
    )
    overlap_threshold = st.sidebar.slider("Overlap_threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


if __name__ == "__main__":
    main()
