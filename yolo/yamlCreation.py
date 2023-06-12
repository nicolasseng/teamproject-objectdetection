import platform
import glob
import os
import yaml

def gatherFolderPath(queriedFolderName: str) -> str:
    """
    Takes a string denoting the folder name and returns the absolute path to the folder if found.
    Otherwise, returns "none" as a string.

    The function searches relative to the root of the environment.
    If searching for a folder within one (or multiple) directories, add **/ to the query.

    Example usage:
    gatherFolderPath("data")  # Searching for a folder named "data"
    gatherFolderPath("**/images")  # Searching for a folder named "images" within any subdirectory
    """

    if platform.system() == "Windows":
        # Running on Windows
        convertedQueryPath = "**\\" + queriedFolderName
    else:
        # Running on Unix (including macOS and Linux)
        convertedQueryPath = "**/" + queriedFolderName

    try:
        # Selecting and gathering all directories matching the queried folder name
        listOfFolders = glob.glob(convertedQueryPath, recursive=True)

        # Returning the first result
        # Should be the only one
        if listOfFolders:
            return os.path.abspath(listOfFolders[0])
        else:
            return "none"
    except:
        return "none"
    
    
def createPath(rootPath, folder):
    if platform.system() == "Windows":
        # Running on Windows
        convertedQueryPath = rootPath + "\\" + folder
    else:
        # Running on Unix (including macOS and Linux)
        convertedQueryPath = rootPath + "/" + folder

    return convertedQueryPath


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
