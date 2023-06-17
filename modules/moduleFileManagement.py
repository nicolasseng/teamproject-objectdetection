'''
<not sure whether we should refactor this much>
This file contains several functions to manage, access, read or modify files
'''
# --- / 
# -- / external imports 
import glob 
import platform
import os
from typing import Optional

from numpy import log

# --- /
# -- / function to read a given file
# returns "file not found" if no file was found  
def readFile(filepath:str) -> Optional[str]:
    '''
    takes filepath as argument **string** 
    reads out given file and returns its content as string 
    returns None if path was not found
    
    ### Example usage: 
    readFile(README.MD) -> will return the whole file as one string, with "\\n" etc
    '''
    try:
        with open(filepath,'r') as fileObj : 
            readFile = fileObj.read()
    except:
        readFile = None
    return readFile;


def gatherFolderContent(queriedFolderPath:str) -> Optional[list]:
    ''' 
    function takes a **string** denoting a path and returns a list with all of its items 
    
    ### example usage: 
    gatherFolderContent("sample_img") -> List [ "path/sampleImg.jpg",...,"path/sampleImg4.jpg"]
    ''' 
    folderPath:str =gatherFolderPath(queriedFolderPath)
    globQuery:str = "{}{}*".format(folderPath,os.sep)
    listOfFiles = glob.glob(globQuery)
    
    return listOfFiles
    
# --- / 
# -- / 
def gatherFilePath(queriedFilename:str) -> Optional[str]: 
    ''' 
    takes a string denoting filename, returns path of it, if found 
    otherwise returns None
    
    Function searches relative from root of environment. 
    if searching for a file within 1 (or multiple) directories, add **/ to the query.
    
    ### Example usage: 
    gatherFillePath("**/randomfile.txt") --> can be located in a diretory like: /path/to/file/1/2/randomfile.txt
    gatherFilePath("**/MobileNetSSD_deploy") --> searching for file data/preTrainedMSSD/*
    gatherFilePath("*.py) -> returns first python file found in root directory
    '''
    convertedQueryPath = "**/"+queriedFilename
    try:
        # selecting and gathering all files in 
        listOfFiles = glob.glob(queriedFilename,recursive=True)
        return listOfFiles[0];
    except: 
        print("no file found with given query : {}".format(queriedFilename))
        return None 
        
# --- / 
# -- / 
def gatherFolderPath(queriedFolderName: str) -> Optional[str]:
    """
    Takes a string denoting the folder name and returns the absolute path to the folder if found.
    Otherwise, returns None  

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
            return None
    except:
        return None

def createPath(rootPath:str, folder:str) -> str:
    if platform.system() == "Windows":
        # Running on Windows
        convertedQueryPath = rootPath + "\\" + folder
    else:
        # Running on Unix (including macOS and Linux)
        convertedQueryPath = rootPath + "/" + folder

    return convertedQueryPath


if __name__ == "__main__":
    exit("not meant to be run")