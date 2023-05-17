'''
<not sure whether we should refactor this much>
This file contains several functions to manage, access, read or modify files
'''

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

def gatherFileType(filepath:str)-> str:
    #TODO should return more definite type, like Enum 
    pass

if __name__ == "__main__":
    exit("not meant to be run")