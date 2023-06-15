'''
File contains basic structure of Ui, which is supplied by Streamlit 

'''

# --- / 
# -- / external imports 
from streamlit_option_menu import option_menu

# --- / 
# -- / internal imports 
import settings.uiSettings as UiSettings
from modules.moduleShowReadMe import DisplayReadme
from modules.moduleShowSourceCode import ShowSourceCode
from UI.uiRunningApp import run_the_app
from yolo.yolov8_detection import run_yolov8


# --- / 
# -- / function to select mode to display at runtime
def SelectProgramMode()  : 
    '''
    function displaying a streamlit menu with several options
    upon selection of a module to execute, said module is executed 
    '''
    appModeSelection = ["Readme File", "Run SSD", "Yolov8", "Show the Code"];
    appModeSelectionIcons = ["book", "display", "display", "cloud-download"]
    app_mode = option_menu(
        None, 
        appModeSelection,
        orientation="horizontal",
        icons=appModeSelectionIcons, menu_icon="cast"
    ) 
    # selecting appropiate mode: 
    if app_mode == appModeSelection[0]:
        # display readme 
        DisplayReadme();
    elif app_mode == appModeSelection[1]: 
        # runObjectDetection using SSD
        run_the_app()
    elif app_mode == appModeSelection[2]:
        # runObjectDetection using yolov8
        run_yolov8()
    elif app_mode == appModeSelection[3]:
        # showing source code for 
        ShowSourceCode()

# --- / 
# -- / 

if __name__ == "__main__":
    exit("not to be run, use main.py in root")