'''
This file contains all necessities for running the **readme** module of our webapplication 
'''

# --- / 
# -- / external imports 
import streamlit as st 
import markdown 
import os 


def ReadOutMarkdown(file:str) -> str:
    '''
    function returning 
    ''' 
    with open(file,"r") as file: 
        markdownString = file.read();
        
    return markdownString;
    
        
        

def DisplayReadme(): 
    '''
    function showcases Readme
    '''
    filepath = os.path.join(os.getcwd(),"README.md")
    # styling application
    st.title("See our Readme file")
    st.markdown("Below you can have an insight on our readme which contains information on how to run this system and how it is structured")
    st.markdown(ReadOutMarkdown(filepath))

if __name__ == "__main__": 
    # testpath = os.path.join( os.getcwd(), "README.md")
    # print(ProcessReadme(testpath))
    exit("not meant to be run")