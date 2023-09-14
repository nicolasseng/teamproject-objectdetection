'''
This file contains the logic to visualize object detection on a given image  
'''

# --- / 
# -- / external imports
from typing import Callable, Optional
import numpy
import streamlit as st 

# --- /
# -- / internal imports
from modules.moduleFileManagement import prepareImageToSave, saveEvaluationToFile


# --- /
# -- /
def interfaceImage(loadedModel:object,functionRunModel:Callable ,objectClasses:list,selectedImage:str| numpy.ndarray,confidence:float,usedModel:str):
    ''' 
    function that displays both the unprocessed and processed image after being run on a given model 
    takes the following arguments: 
    - loaded Model necessary for the function to run on 
    - functionRunModel with the following signature ( loadedModel:object, confidence:float, objectClasses:list, selectedImage:array|str) 
    - objectClasses :list 
    - selectedImage: numpy. array | str
    - confidence : float
    - usedModel : string representation of used model
    '''
    # once image file was loaded or not 
    col1, col2 = st.columns(2)
    with col1:
        st.image(selectedImage, caption="Selected Image", use_column_width=True)
    with col2:
        
        # varies based on given function
        detectionResult = functionRunModel(loadedModel,confidence,objectClasses,selectedImage)
        # runYoloOnImage(loadedModel,confidence, objectClasses,selectedImage)
        # wrapperRunningDnn(loadedNet,confidence_threshold,objectClasses,selectedImage)

        st.image(detectionResult['image'], caption="Detected Image",
                    use_column_width=True)
        
        # - logic for evaluating results
        dictOfEvaluation:Optional[dict] = formEvaluateResult()
        if dictOfEvaluation == None: 
            return 
        
        # collecting all results: 
        imageListRepresentation:Optional[list] = prepareImageToSave(detectionResult["image"])
        imageDimension:tuple = detectionResult["image"].shape
        
        dictDetectionEval:dict= {
            "usedModel": usedModel,
            "confidence": confidence,
            "amountDetected": len(detectionResult["foundObjects"]),
            "actualAmount": dictOfEvaluation["amount"],
            "faultyDetection": dictOfEvaluation["faulty"],
            "imageArray": imageListRepresentation,
            "imageDimension": imageDimension
            }
        saveEvaluationToFile(dictDetectionEval,dictDetectionEval["usedModel"])
    


# dictionary returned should contain: 
# - usedModel -> to be displayed upon showcase
# - confidence level : 
# - detected amount of objects
# - actual amount of objects detected -> user input!
# - amount of faulty detection -> user input!
# - image encoded as 1D-array 
# - image dimensions used to reshape accordingly
# --- / 
# -- / 
def formEvaluateResult()-> Optional[dict]: 
    '''
    function that is providing UI to submit evaluation to store 
    This function is used to display a small form that takes information after running object detection. 
    It only returns a dictionary if the button was pressed on the webinterface and none otherwise 
    '''
    with st.form("evaluating results"):
        st.write("insert the correct data for this detected image")
        st.write("once done, click the submit-button to save the result")
        actualDetection:int = int(st.number_input("actual amount of detections",min_value=0,max_value=20,step=1))
        amountFaultyDetection:int = int(st.number_input("amount of faulty detections",min_value=0,max_value=20,step=1))
        submitted = st.form_submit_button("save results")
        if submitted:
            resultDict:dict = {
                "amount": actualDetection,
                "faulty": amountFaultyDetection
            }
            return resultDict
        return None