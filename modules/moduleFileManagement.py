'''
<not sure whether we should refactor this much>
This file contains several functions to manage, access, read or modify files
'''
# --- / 
# -- / external imports 
import glob 
import os

# --- /
# -- / function to read a given fiel
# returns "file not found" if no file was found  
def readFile(filepath:str) -> str:
    '''
    '''
    try:
        with open(filepath,'r') as fileObj : 
            readFile = fileObj.read()
    except:
        readFile = "file not found"
    return readFile;

def gatherFileType(filepath:str): #-> str:
    #TODO should return more definite type, like Enum 
    pass

def gatherFilePath(queriedFilename:str) -> str: 
    ''' 
    takes a string denoting filename, returns path of it, if found 
    otherwise returns "none" as string
    
    Function searches relative from root of environment. 
    if searching for a file within 1 (or multiple) directories, add **/ to the query.
    ### Example usage: 
    gatherFilePath("**/MobileNetSSD_deploy") --> searching for file data/preTrainedMSSD/*
    gatherFilePath("*.py) -> returns first python file found in root directory
    '''
    # TODO write correctly, according error, if not found ! \
    convertedQueryPath = "**/"+queriedFilename
    try:
        # selecting and gathering all files in 
        listOfFiles = glob.glob(queriedFilename,recursive=True)
        
        # returning first result 
        # should be the only one TODO test and improve result, pls! 
        return listOfFiles[0];
    except: # likely nothing find -->  TODO: indicate missing file
        return "none" # Really bad, TODO: I will focus on it later 
        

if __name__ == "__main__":
    exit("not meant to be run")