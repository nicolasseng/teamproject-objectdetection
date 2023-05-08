import urllib
from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Object Detection Web App", layout="wide")

# HEADER
st.subheader("Object Detection Web App")
confidence = 0.25
readme_text = None


def main():
    # st.sidebar.title("What to do?")
    app_mode = option_menu(
        None,
        ["Readme File", "Run Application", "Show the Code", "Upload a File?"],
        orientation="horizontal",
        icons=["book", "display", "download", "cloud-upload"],
    )
    if app_mode == "Run Application":
        run_the_app()
    elif app_mode == "Readme File":
        readme_text = st.markdown(get_file_content_as_string("README.md"))
    elif app_mode == "Show the Code":
        # readme_text.empty()
        st.code(get_file_content_as_string("FirstInterface.py"))
    elif app_mode == "Upload a File?":
        upload_file()


def upload_file():
    uploaded_file = st.file_uploader("Choose a file:")
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
        st.info("File upload was successfull", icon=ℹ️)


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
        st.info("File upload was successfull", icon=ℹ️)


@st.cache_resource(show_spinner=False)
def get_file_content_as_string(path):
    url = (
        "https://raw.githubusercontent.com/nicolasseng/teamproject-objectdetection/FileUpload/"
        + path
    )
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def run_the_app():
    st.sidebar.markdown("# Settings")
    model_src = st.sidebar.radio(
        "Select file:", ["Use our demo files", "Use your own files"]
    )
    if model_src == "Use your own files":
        upload_file_sidebar()
        col1, col2 = st.columns(2)
        with col1:
            image2 = Image.open("data/sample_img/Image3.jpeg")
            st.image(
                image2,
                use_column_width="auto",
                caption="Image without object detection",
            )
        with col2:
            image = Image.open("data/sample_img/Test_Image.png")
            st.image(
                image, use_column_width="auto", caption="image with object detection"
            )
    elif model_src == "Use our demo files":
        col1, col2 = st.columns(2)
        with col1:
            image2 = Image.open("data/sample_img/Image3.jpeg")
            st.image(
                image2,
                use_column_width="auto",
                caption="Image without object detection",
            )
        with col2:
            image = Image.open("data/sample_img/Test_Image.png")
            st.image(
                image, use_column_width="auto", caption="image with object detection"
            )

    st.progress(80, "Intersection over Union:")

    input_option = st.sidebar.radio(
        "Select input type: ", ["image", "video", "livestream"]
    )

    model_names = list({"bikes", "car", "Traffic light", "human"})
    if st.sidebar.checkbox("Search for specific objects?"):
        assigned_class = st.sidebar.multiselect(
            "Select objects", model_names, default=[model_names[0]]
        )
        classes = [model_names.index(name) for name in assigned_class]

    # selected_frame_index = st.sidebar.slider(
    #     "Switch between your images", min_value=0, max_value=5, value=0, step=1
    # )
    st.sidebar.image(image2)
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", min_value=0.1, max_value=1.0, value=0.45
    )


if __name__ == "__main__":
    main()
