'''
This file contains logic to display the source code of a certain files with streamlit 
author = Fabian S. 
'''

# --- / 
# -- / external imports
import streamlit as st
import glob 
import os 

# --- / 
# -- / internal imports
from modules.moduleFileManagement import readFile
# --- / 
# -- / 
def ShowSourceCode() -> None:
    '''
    TODO refactor accordingly
    TODO include dropdown for all available folders
    function to display selection of sourcecode 
    
    creates a streamlit page displaying selected code within a codeblock
    '''
    # creating query for each file that ends with .py 
    basePath: str = "**/*.py"
    ListOfPythonFiles = glob.glob(basePath,recursive=True)
    # setting up logic for parsing file to string and showcasing it 
    quantityOfFiles = ListOfPythonFiles.__len__()
    
    st.title("Explore Source Code of this project")
    st.markdown("below you can find a selection of all available files in the current direction")
    st.markdown("the directory can be changed at a later point, this is still work in progress")
    st.markdown("**{} Files** are available".format(quantityOfFiles))
    st.write("current directory: {}".format(os.getcwd()))
    indexSelection:int = 0;
    # reacting to requests to change selected code
    if 'indexSelection' not in st.session_state:
        st.session_state.indexSelection = indexSelection
    if st.button("next file") and (st.session_state.indexSelection < quantityOfFiles-1):
        st.session_state.indexSelection += 1 
    if st.button("previous file"):
        if st.session_state.indexSelection > 0:
            st.session_state.indexSelection -=1

    # displaying current selection 
    indexSelection = st.session_state.indexSelection
    WrapCodeBlock(ListOfPythonFiles[indexSelection])
    
def WrapCodeBlock (filepath:str):
    '''
    function taking filepath and creating a streamlit **code block** with this files content 
    returns nothing
    '''
    sourceCode:str = readFile(filepath)
    st.markdown("### {}".format(filepath))
    st.code(sourceCode,language='python')
    
    
if __name__ == "__main__":
    
    exit("not meant to be run, module package")