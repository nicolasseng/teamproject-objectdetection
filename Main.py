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
    SelectProgramMode();




if __name__ == "__main__":
    main()