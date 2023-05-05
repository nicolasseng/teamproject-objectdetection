import urllib
from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Object Detection Web App", layout="wide")

# HEADER
st.subheader("Object Detection Web App")
model = None
confidence = 0.25

st.write("Open the Sidebar to either upload a file or show the source code.")


def main():
    readme_text = st.markdown(get_file_content_as_string("README.md"))
    st.sidebar.title("What to do?")
    app_mode = st.sidebar.selectbox(
        "What do you want to do?",
        ["Home", "Run the App", "Show the Code", "Upload a File"],
    )
    if app_mode == "Run the App":
        run_the_app()
    elif app_mode == "Show the Code":
        readme_text.empty()
        st.code(get_file_content_as_string("FirstInterface.py"))
    elif app_mode == "Upload a File":
        upload_file()


def upload_file():
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


@st.cache_resource(show_spinner=False)
def get_file_content_as_string(path):
    url = (
        "https://raw.githubusercontent.com/nicolasseng/teamproject-objectdetection/main/"
        + path
    )
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def run_the_app():
    st.sidebar.markdown("# Settings")
    model_src = st.sidebar.radio(
        "Select file", ["Use our demo files", "Use your own files"]
    )
    if model_src == "Use your own files":
        upload_file()
    elif model_src == "Use our demo files":
        image = Image.open("data/sample_img/Test_Image.png")
        st.image(image, caption="This is an example image")

    input_option = st.sidebar.radio(
        "Select input type: ", ["image", "video", "livestream"]
    )

    model_names = list({"bikes", "car", "Traffic light", "human"})
    if st.sidebar.checkbox("Search for specific objects?"):
        assigned_class = st.sidebar.multiselect(
            "Select objects", model_names, default=[model_names[0]]
        )
        classes = [model_names.index(name) for name in assigned_class]

    selected_frame_index = st.sidebar.slider(
        "Choose an image (index)", min_value=0, max_value=5, value=0, step=1
    )
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", min_value=0.1, max_value=1.0, value=0.45
    )


if __name__ == "__main__":
    main()
