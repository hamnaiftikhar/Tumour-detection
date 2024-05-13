# Tumour-detection using pre trained resnet-152

Objective:
The objective of this report is to document the steps involved in preparing image data for tumour
classification using machine learning techniques.


Step 1: Data Collection:
Two directories were identified for data collection:
 Folder containing images of "No tumour”.
 Folder containing images of "tumour”.


Step 2: Data Inspection:
Using the `os` module, the number of images in each directory was obtained. This was done to
understand the distribution of data between the two classes.


Step 3: Labelling:
The images were labelled according to their respective classes:
 "No tumour" was labelled as 0
 "tumour" was labelled as 1


Step 4: Image Preprocessing:
 Images were converted into NumPy arrays.
 They were resized to dimensions of 224 x 224 x 3 to fit the input requirements of the deep
learning model.
 Colour mode was converted to RGB.
 The images were then appended to a list called `data`.


Step 5: Train-Test Split:
Using the `train_test_split` function from scikit-learn, the dataset was split into training and testing
sets. The testing set size was set to 10% of the total data, and shuffling was enabled.


Step 6: Data Visualization:
A sample of the training data was visualized using matplotlib. A grid of 4x4 subplots was created to
display 16 randomly selected images along with their corresponding class labels ("No tumour" or
"tumour"). 
