"""
file takng care of creation of Yaml files which are required for succesfully training a yolo model. 
Further modifications or adaptions for Yaml-Files may be added later
"""

# --- / 
# -- / external imports 
import os
from typing import Optional
import yaml
# --- / 
# -- / internal imports 
from modules.moduleFileManagement import createPath, gatherFolderPath

def createYaml():
    '''
    searches for folders with name "test" and "valid" 
    if found: creates yaml file which contains those both folders as source for training datasets 
    if not : raises Exception "no Path was found, aborting" 
    '''
    try: 
        testPath = gatherFolderPath('test')
        validPath = gatherFolderPath('valid')
            
        config = {
            'train': createPath(testPath, 'images'),
            'val': createPath(validPath, 'images'),
            'nc': 5,
            'names': ['Helmet', 'Goggles', 'Jacket', 'Gloves', 'Footwear']
        }
        # Specify the subfolder name
        subfolder = 'yolo'
        # Create the subfolder if it doesn't exist
        os.makedirs(subfolder, exist_ok=True)
        
        # Construct the file path within the subfolder
        
        file_path = os.path.join(subfolder, 'yolov8_config.yaml')

        with open(file_path, 'w') as file:
            yaml.dump(config, file, sort_keys=False)
    except:
        raise Exception("no Path was found, aborting")

if __name__ == "__main__":
    exit("not meant to be run")