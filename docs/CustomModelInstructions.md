# Custom Model Training Instructions

Follow these steps to train a custom model using the YOLOv8 algorithm.

## Step 1: Prepare Training Data
1. If you want to use the provided dataset:
   - In the `\data\TrainingYolo\TrainingDataYolo` directory of your project repository, locate the zipped folder named 'data'.
   - Unzip the folder to extract the images and labels required for training.

   Alternatively, if you want to use your own dataset:
   - Prepare your own dataset of images and corresponding labels that you want to use for training.
   - Ensure that your dataset follows the required format for YOLOv8 training.

## Step 2: Replace Training Data (if using your own dataset)
1. If you're using the provided dataset, skip this step.
2. If you're using your own dataset, replace the existing zipped folder named 'data' in the `\data\TrainingYolo\TrainingDataYolo` directory of your project repository with your own dataset.
   Note: Make sure your dataset is in the same format as the provided sample

## Step 3: Select YOLOv8 in Streamlit Application
1. Open the Streamlit application.
2. Look for the option to choose the YOLOv8 algorithm.
3. Select YOLOv8 as the chosen algorithm for training.

## Step 4: Choose Offline Data
1. Locate the sidebar in the Streamlit application.
2. Find and select the option for 'Offline Data'.

## Step 5: Start Model Training
1. Click on the 'Train Model' button to initiate the training process.
2. The training process may take a while, depending on your hardware capabilities.
3. Once the training is completed, a new folder named 'runs' should be created.

## Step 6: Select 'Custom' Model
1. Find the 'Select YOLOv8 Model' dropdown menu.
2. Choose the 'Custom' option from the dropdown menu.
   Note: Selecting this option before training your own model will result in an error.

Congratulations! You have successfully trained a custom model using the YOLOv8 algorithm.