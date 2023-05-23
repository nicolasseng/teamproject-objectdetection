import urllib
from io import StringIO

# import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
# import torch
from PIL import Image
from streamlit_option_menu import option_menu

# --- / 
# -- / internal imports 
from UI.uiFoundation import SelectProgramMode

st.set_page_config(page_title="Object Detection Web App", layout="wide")

# HEADER
st.subheader("Object Detection Web App")
confidence = 0.25
readme_text = None


def main():
    # st.sidebar.title("What to do?")
    SelectProgramMode();
    # app_mode = option_menu(
    #     None,
    #     ["Readme File", "Run Application", "Show the Code", "Upload a File?"],
    #     orientation="horizontal",
    #     icons=["book", "display", "download", "cloud-upload"],
    # )
    # if app_mode == "Run Application":
    #     run_the_app()
    # elif app_mode == "Readme File":
    #     readme_text = st.markdown(get_file_content_as_string("README.md"))
    # elif app_mode == "Show the Code":
    #     # readme_text.empty()
    #     st.code(get_file_content_as_string("FirstInterface.py"))
    # elif app_mode == "Upload a File?":
    #     upload_file()


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



if __name__ == "__main__":
    main()