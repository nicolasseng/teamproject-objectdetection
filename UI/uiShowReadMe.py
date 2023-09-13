'''
This file contains the UI element for displaying the **readme** module of our webapplication
'''

# --- / 
# -- / external imports 
import streamlit as st 
import markdown

# --- / 
# -- / internal imports 

from modules.moduleFileManagement import gatherFilePath, readFile
        
@st.cache_resource(show_spinner=False)

# --- / 
# -- / function to display readme 
def DisplayReadme() -> None: 
    '''
    function querying for "README.md" opening and displaying it with streamlit
    '''
    filepath: str | None = gatherFilePath("README.md")
    if filepath == None: 
        exit()
    # styling application
    st.title("See our Readme file")
    st.markdown("Below you can have an insight on our readme which contains information on how to run this system and how it is structured")
    st.markdown(readFile(filepath))