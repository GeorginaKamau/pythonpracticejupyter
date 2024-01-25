#import libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight




#load and preprocess data
labels =["cats", "dogs", "others"]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(labels)
# Load and preprocess data
def load_and_preprocess_data(base_path):
    labels = []
    images = []

    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img / 255.0
                images.append(img)  # Append the preprocessed image to the list
                labels.append(label)

    unique_labels = set(labels)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels, num_classes=len(unique_labels))

    X_train, X_temp, y_train, y_temp = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return (
        np.array(X_train),
        np.array(X_val),
        np.array(X_test),
        np.array(y_train),
        np.array(y_val),
        np.array(y_test),
        label_encoder
    )


base_path = 'data/train'
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_preprocess_data(base_path)

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Define a learning rate schedule
def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr


# Create a learning rate scheduler
lr_schedule = LearningRateScheduler(step_decay)



#define the model - Convulutional NN
model = models.Sequential()
#building the NN layer by layer
#32 filters of size (3, 3)ReLu activation function
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape=(64, 64, 3))) #3D  #3D 64* 64 pictures
#add 1st layer--filters increased by 64
model.add(layers.BatchNormalization())  # Add Batch Normalization after the convolutional layer
model.add(layers.MaxPooling2D((2, 2)))
#add 2nd layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # Add Batch Normalization after the convolutional layer
model.add(layers.MaxPooling2D((2, 2)))
#add 3rd layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # Add Batch Normalization after the convolutional layer
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())  # Add Batch Normalization after the convolutional layer
model.add(layers.MaxPooling2D((2, 2)))
#output flattened from 3D into 1D array
model.add(layers.Flatten())
#adds fully connected layer with 128 neurons
model.add(layers.Dense(128, activation ='relu'))
model.add(layers.BatchNormalization())  # Add Batch Normalization after the convolutional layer
model.add(layers.Dropout(0.5))  #dropout layer for regularization

model.add(layers.Dense(128, activation ='relu'))
model.add(layers.BatchNormalization())  # Add Batch Normalization after the convolutional layer
model.add(layers.Dropout(0.5))

#final output layer with multiple neurons and softmax activation function
model.add(layers.Dense(num_classes, activation = 'softmax'))

#compile the model
#Adam optimization algorithm - it adapts learning rates for each parameter based on their past gradient
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy']
)


# Define early stopping callback --using EarlyStopping technique in Keras--stops when the validation loss stops improving and restores the model weights from the epoch with the best values of monitored metric
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compute class weights
class_counts = np.sum(y_train, axis=0)
total_samples = len(y_train)
class_weights = total_samples / (len(np.unique(y_train, axis=0)) * class_counts)
class_weight_dict = dict(enumerate(class_weights))


# Use class weights in the fit function
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=50,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,  # Apply class weights here
    callbacks=[lr_schedule],
)


# Save the entire model
model.save('C:/Users/GKamau/PycharmProjects/CNN/cnn_model1.h5')


#stopped at epoch 13, loss=0.0803, accuracy = 0.9688, val_loss=0.7012, val_accuracy=0.8190

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(datagen.flow(X_test, y_test, batch_size=32))
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Function to make a prediction on a single image
def make_prediction(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)

    #convert prediction to class label
    predicted_class = np.argmax(prediction)

    # Check with ; 0 or 1
    if np.argmax(prediction) == 0:
        label = "This is a Cat"
    elif np.argmax(prediction) == 1:
        label = "This is a Dog"
    else:
     label = "This is neither a cat nor a dog"

    print("Prediction Probability:", prediction)
    return label


# Path to the image you want to predict
image_path_to_predict = 'C:/Users/GKamau/PycharmProjects/CNN/data/test/cats/cat.128.jpg'

# Make a prediction using the defined function
make_prediction(model, image_path_to_predict)








