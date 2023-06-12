

# --- / 
# -- / external imports 
import os
import yaml
# --- / 
# -- / internal imports 
from modules.moduleFileManagement import createPath, gatherFolderPath

def createYaml():
    config = {
        'train': createPath(gatherFolderPath('test'), 'images'),
        'val': createPath(gatherFolderPath('valid'), 'images'),
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
