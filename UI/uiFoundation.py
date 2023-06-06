'''
File contains basic structure of Ui, which is supplied by Streamlit 

'''

# --- / 
# -- / external imports 
from streamlit_option_menu import option_menu as stOptionMenu

# --- / 
# -- / internal imports 
import settings.uiSettings as UiSettings
from modules.moduleShowReadMe import DisplayReadme
from modules.moduleShowSourceCode import ShowSourceCode
from UI.uiRunningApp import run_the_app
from yolo.yolov5_detection import run_yolo
from yolo.yolov8_detection import run_yolov8


# --- / 
# -- / function to select mode to display at runtime
def SelectProgramMode()  : 
    '''
    function displaying a streamlit menu with several options
    upon selection of a module to execute, said module is executed 
    '''
    appModeSelection = ["Readme File", "Run SSD", "Run Yolo", "Yolov8", "Show the Code", "Upload File[not used]"];
    appModeSelectionIcons = ["book", "display", "download", "cloud-upload"],
    app_mode = stOptionMenu(
        None, 
        appModeSelection,
        orientation="horizontal",
        icons=appModeSelectionIcons,
    ) 
    # selecting appropiate mode: 
    if app_mode == appModeSelection[0]:
        # display readme 
        DisplayReadme();
    elif app_mode == appModeSelection[1]: 
        #runObjectDetection 
        run_the_app()
    elif app_mode == appModeSelection[2]:
        run_yolo()
    elif app_mode == appModeSelection[3]:
        run_yolov8()
    elif app_mode == appModeSelection[4]:
        # showing source code for 
        ShowSourceCode()
    elif app_mode == appModeSelection[5]:
        #uploading files 
        exit()

# --- / 
# -- / 

if __name__ == "__main__":
    exit("not to be run, use main.py in root")