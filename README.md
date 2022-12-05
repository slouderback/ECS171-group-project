# ECS171 Final Group Project: Pizza/Not Pizza Image Classifier
Data set used for the project: https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza

## Introduction
For our final project, we decided to build an image classifier that determines whether a picture is of pizza or a different food. We selected this topic because it seemed fun and interesting, and everyone likes pizza. Also, the task of building the model seemed very manageable given the tools that we had learned throughout this course. We also felt that image classifiers were one of the more fascinating topics and this project allowed us to build a complete model on our own.

Building good predictive models can have larger impacts than we anticipate. While our model only works on pizza, it can easily be tweaked to a different image classifier by changing the input data. It can also be expanded upon by adding different labels for new categories of foods. In some cases, it can even be applied in unrelated topics, such as Hisashi Kambe's bakery image classifier that went on to be used for cancer identification!

## Methods

### Data Evaluation
Data evaluation was performed mostly via importing all images to a Keras dataset via the provided library function `image_dataset_from_directory`, and analyzing characteristics of the data with the outputted dataset. Manual review of all images was done before import to remove outstanding outliers (mostly photos in which the primary subject of the shot was not the food itself, such as with the pizza directory).

### Pre-Processing
The dataset consists of 983 images of pizza and 983 images of non-pizza foods - four outliers were culled, so we instead have 979 pizza images and 983 non-pizza food images. All images were resized by the provider to have either a width or height dimension of 512 pixels, with a varying dimension for the other dimension. While the Keras import will automatically resizes all images to 256x256, we additionally chose to crop all of the images such that we have uniform 1:1 square photos that are focused on the subject foods of each image via the Keras hyperparameter `crop_to_aspect_ratio`.

<img src="notebook_resources/pizza.jpg" alt="notebook_resources/pizza.jpg" title="Pizza and Not Pizza Images" width="480"/>
Figure 1: Pizza and Not Pizza Images  <br />

For models 1 and 3, we kept the images as a resolution of 256x256 and RGB.

<img src="notebook_resources/secondmodel_pizzaimages.png" alt="notebook_resources/secondmodel_pizzaimages.png" title="Gray-Scaled Pizza and Not Pizza Images" width="480"/>
Figure 2: Gray-Scaled Pizza and Not Pizza Images  <br />

For model 2, we kept the images at a resoltuion of 256x256, but we gray-scaled the images. 

<img src="notebook_resources/fourthmodel_pizzaimages.png" alt="notebook_resources/fourthmodel_pizzaimages.png" title="Gray-Scaled Pizza and Not Pizza Images" width="480"/>
Figure 3: Lower Resolution Pizza and Not Pizza Images  <br />

For model 4, we kept the RGB of the images but downscaled the resolution to 64x64. 

All image pixel values will thus need to be normalized before being input to our model CNN — we implement this via a normalization/standardization layer implemented in our model such that all image data input is automatically normalized before entering the actual CNN. Per the documentation, image flattening is not particularly necessary here.

### First Model Pass
The first model is an initial test model based on previous homeworks and examples from TensorFlow documentation on implementation of a basic convolution neural network without any data augmentation, primarily using ReLu activations (with one final sigmoid activation for binary classification), with the Adam optimizer and binary cross-entropy as our loss model. 

```
model = tf.keras.Sequential([
  normalization_layer, # The normalizer/standardizer layer from pre-processing section
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = (256, 256, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# From the referenced Medium article, use the Adam optimizer 
# and binary cross-entropy, since we're making a binary classifier.
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.BinaryCrossentropy(),
  metrics=['accuracy'])
```

### Second Model Pass

The second model is one of the new models that we built to test against our first model. One of the changes from our original model was the actual data. For this model, we gray scaled each of the pizza images. 

```
# Convert to grayscale (a float between 0-1) images and resize to 256x256 (default)
def load_images(filepath, size = (256, 256)):
    return np.asarray(Image.open(filepath).resize(size).convert('L')) / 255.0
```

Another change was to the convolutional neural network model itself. We removed three hidden layers: two Conv2D layers and one MaxPooling2D layer. 

```
model2 = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
```

Besides the removal of these layers, the model has the same hidden layers with the same activation functions. The model is compiled the same as the first model with the Adam optimizer and binary cross-entropy loss model. 

### Third Model Pass

For the third model, we added new hidden layers, reduced the number of neurons per layer, and added a dropout layer. We added the dropout layer in between the first Dense layer. In addition, for each dense layer, we reduced the number of neurons from 128 to 16. Finally, we added two more dense layers each with a 'relu' activation function. 

```
model3 = tf.keras.Sequential([
  normalization_layer, # The normalizer/standardizer layer from pre-processing section
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = (256, 256, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1, activation="sigmoid")
])
```

The model is compiled the same way as models 1 and 2 with the Adam optimizer and the binary cross-entropy loss function.

### Fourth Model Pass

For the final model, we made changes both to the data but also to the neural network. For the data, we downscaled the resolution from 256x256 to 64x64.

```
# This time, crop images to 64x64
finalData = tf.keras.utils.image_dataset_from_directory(
    "pizza_not_pizza",
    labels='inferred',
    label_mode='binary',
    color_mode='rgb',
    image_size=(64, 64),
    crop_to_aspect_ratio=True)
```

For the neural network model, we simplified everything about the model including the number of layers and number of neurons per layer. We removed a MaxPooling2D layer and Conv2D layer as well as reduced the number of neurons in the 'relu' dense layer. In addition, we reduced the kernel size of both of the Conv2D layer as well as adjusted the input_shape of the first Conv2D layer in accordance to the resolution change of input images explained above. 

```
modelF = tf.keras.Sequential([
    normalization_layer_64,
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape = (64, 64, 3)),
    tf.keras.layers.MaxPooling2D((8, 8)),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

The model is compiled the same way as models 1, 2, and 3 with the Adam optimizer and the binary cross-entropy loss function.

## Results
<img src="notebook_resources/model-fit-graph.png" alt="notebook_resources/model-fit-graph.png" title="Model Fit Graph"  width="480"/>
Figure 4: Model Fit Graph  
<img src="notebook_resources/firstmodel_trainingresults.jpg" alt="notebook_resources/firstmodel_trainingresults.jpg" title="First Model Training Results"  width="480"/>
Figure 4: First Model's Training Results  
<img src="notebook_resources/firstmodel_testingresults.jpg" alt="notebook_resources/firstmodel_testingresults.jpg" title="First Model Testing Results" width="480"/>
Figure 5: First Model's Testing Results  

It expectedly has very good accuracy and loss metrics in training, but has conversely inadequate accuracy and loss when using testing data. Potential updates to improve this model would likely include data augmentation to diversify the range of data the model gets, internal changes to the model itself (such as with different hidden layers, activation functions, or hyperparameters), and/or inclusion of a validation data split (natively supported by Keras) to monitor the model's performance with non-training data.

## Discussion

## Conclusion

## Collaboration


