import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Set up your dataset path
dataset_path = 'flowers'  # Update this path

# Load the dataset using ImageDataGenerator
batch_size = 64
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Add validation_split

# Use flow_from_directory to load the images from the directory
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'  # Set as training data
)

# Create a validation generator
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'  # Set as validation data
)

# Getting number of classes
num_classes = train_generator.num_classes

# Load the SimCLR model as a Keras layer
hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'
module = hub.KerasLayer(hub_path, input_shape=(224, 224, 3), trainable=False)

# Define the fine-tuning process
def create_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base_model = module(inputs)  # Call the Keras layer with inputs
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(base_model)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
total_iterations = 10
history = model.fit(train_generator, 
                    validation_data=validation_generator,  # Include validation data
                    epochs=total_iterations)

# Save the model
model.save('model/simclr_preTrain_flower.h5')  # Save in HDF5 format
print("Model saved as model/simclr_preTrain_flower.h5")

# To calculate accuracy, you can evaluate the model on the training set or a separate validation set.
accuracy = model.evaluate(train_generator)[1]
print(f"Training Accuracy: {accuracy:.4f}")

# Evaluate on validation set
val_accuracy = model.evaluate(validation_generator)[1]
print(f"Validation Accuracy: {val_accuracy:.4f}")


