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

# --- / 
# -- / function to select mode to display at runtime
def selectProgramMode()  : 
    '''
    '''
    appModeSelection = ["Readme File", "Run Application", "Show the Code", "Upload a File?"];
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
        exit();
    elif app_mode == appModeSelection[2]:
        # showing source code for 
        exit();
    elif app_mode == appModeSelection[3]:
        #uploading files 
        exit();

# --- / 
# -- / 

if __name__ == "__main__":
    exit("not to be run, use main.py in root")