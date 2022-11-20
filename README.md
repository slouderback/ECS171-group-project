# ECS171 Final Group Project: Pizza/Not Pizza Image Classifier
Data set used for the project: https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza

## Data Evaluation
Data evaluation was performed mostly via importing all images to a Keras dataset via the provided library function `image_dataset_from_directory`, and analyzing characteristics of the data with the outputted dataset. Manual review of all images was done before import to remove outstanding outliers (mostly photos in which the primary subject of the shot was not the food itself, such as with the pizza directory).

## Pre-Processing
The dataset consists of 983 images of pizza and 983 images of non-pizza foods - four outliers were culled, so we instead have 979 pizza images and 983 non-pizza food images. All images were resized by the provider to have either a width or height dimension of 512 pixels, with a varying dimension for the other dimension. While the Keras import will automatically resizes all images to 256x256, we may choose to crop all of the images such that we have uniform 1:1 square photos that are focused on the subject foods of each image (Keras provides a method for this on import according to the docs).

Per comments from the dataset provider (and the original dataset this data is sourced from, the food101 set), no preliminary preprocessing has been done on this dataset other than the aformentioned resizing. All image pixel values will thus need to be normalized before being input to our model CNN.