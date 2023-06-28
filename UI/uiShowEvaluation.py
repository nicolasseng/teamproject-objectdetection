'''
file containing UI elements to display resutls of various detections

'''
# --- /
# -- / external imports 
from typing import Optional
import numpy
import streamlit as st  

# --- / 
# -- / internal imports
from modules.moduleFileManagement import convertArrayToImage, convertListToArray, gatherFolderContent, gatherFolderPath, loadFromFile


def displayResultSelection(): 
    ''' 
    function generating UI for showing and examining stored test images
    
    '''
    foundResults:Optional[list] = gatherFolderContent("**/detectionResults")
    if (foundResults == None) or (foundResults == []):
        st.error("It seems no evaluation data was retrieved and saved yet.")
        st.error("please run some object detections and save their results after evaluating them!")
        # st.error("No results were found, sorry")
        return
    
    st.divider()
    st.markdown("## examine previous detections and their evaluation")
    st.markdown("Below you can view all saved object detections of images run on different models.")
    st.markdown("this examination will give you some insight on several aspects:")
    st.markdown("- how many objects were detected by the model")
    st.markdown("- at which confidence")
    st.markdown("- how many objects are actually contained in the image") 
    st.markdown("- how many objects were examined wrong by the model")
    
    if len(foundResults) != 1:
        selectedEvaluation:int = st.slider("Select Source to show", 
                                   min_value=1, 
                                   max_value=len(foundResults),
                                   step=1,
                                  value=0
                                   )
        if selectedEvaluation == None:
            return   
        selectedEvaluation -= 1 
    else: 
        selectedEvaluation=0
    
    st.divider()
    
    selectedPath:str = foundResults[selectedEvaluation]
    try:
        extractedDictionary: Optional[dict] = loadFromFile(selectedPath)
        if extractedDictionary == None:
            raise Exception()
        
        
        col1,col2 = st.columns(2)
        
        with col1:
            #displaying image
            extractedArray:numpy.ndarray = convertListToArray(extractedDictionary["imageArray"])
            extractedDimensions:list = extractedDictionary["imageDimension"]
            convertedImageArray:numpy.ndarray = convertArrayToImage(extractedArray,extractedDimensions)
            st.image(convertedImageArray,"extracted image",use_column_width=True)
            
        with col2: 
            st.markdown("## Evaluation of Object detection:")
            st.markdown("run **model** : {}".format(extractedDictionary["usedModel"]))
            st.markdown("applied **confidence** : {}".format(extractedDictionary["confidence"]) )
            st.markdown("detected objects **by model** : {}".format(extractedDictionary["amountDetected"]))
            st.markdown("actual amount of objects: {}".format(extractedDictionary["actualAmount"]))
            st.markdown("mistakenly detected Objects: {}".format(extractedDictionary["faultyDetection"]))
            
            
        
        
    except:
        st.error("could not extract information")
        return
        
    
