
# Coding Convetions 
within our project: 

---
We've come up with a little guideline for naming, styling and coding within our environment. 
To give an overview we are structuring our **coding conventions** into _naming things_, _styling things_ and _organizing things_. 

## Naming Things : 

We generally endorse the use of **camelCase** as it provides a good overview and vision to functions and variables. 

this styling is generally used for the following instances:
- functions
- variables 
- methods
- attributes

examples for this styling methodology: 
```Python

def functionDoingSomething(parameterOne:int,parameterTwo:str) -> bool: 
    processedData: int = processData(parameterOne)
    evaluatedData:bool = evaluateInput(processedData,parameterTwo)
    return evaluatedData
```

For classes we deploy **PascalCase**:

```Python
class ThisIsAClass: 
    def __init__(): 
        ...
```

## Styling Things : 
Python is not strong with typing however deploying strict types helps to code with a better clearance.

For that we are endorsing the use of Pythons optional typing hints. 


### Functions/methods: 
We generally **require** the usage of types both for the applied parameters of a function and its return. 

Whenever a function/method may also return an undefined state such as **none** - i.e. an error occured and needs to be signaled or similar - we endorse the usage of the **Optional[type]** type hint. 

Further any function/method may include a **Docstring** containing information such as: 
- accepted/required inputs 
- short example usage 

An example for correct naming of functions would then be: 
```Python 
def multiplyNumbers(firstNumber:int,secondNumber:int) -> int: 
    """
    function takes two integer and returns their product

    ### example usage: 
    multiplyNumbers(1,2) -> 2
    """
    result = firstNumber * secondNumber
    return result
```


## Organizing Things: 

Our code base is structured into the following directories: 
- **data**: containing pre-trained models, sample images, videos or training sets 
- **UI** : containing code used for the ui - primarily streamlit
- **modules** : each file in this directory serves a purpose and provides code for use in other areas of our codebase. 
- **settings** : this directory contains global settings such as filepaths for weights or similar. 
- **docs** : contains documents about several considerations, conventions (such as this) or tutorials for usage

Further we have : 
- README.md : this repository's primary source for giving an overview 
- Main.py : primarily file used for starting the application. 
- pythonDependency.txt : contains a list of all libraries necessary for running this project