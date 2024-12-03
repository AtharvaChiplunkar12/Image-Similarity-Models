import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load CIFAR-10 dataset
(cifar_train_images, cifar_train_labels), _ = tf.keras.datasets.cifar10.load_data()  # Only load training data

# Normalize the images
cifar_train_images = cifar_train_images.astype("float32") / 255.0

# Resize images to 224x224 to match the input requirements of SimCLR
cifar_train_images = tf.image.resize(cifar_train_images, (224, 224))

# Number of classes in CIFAR-10
num_classes = 10

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
history = model.fit(
    cifar_train_images, cifar_train_labels, 
    epochs=total_iterations,
    batch_size=64
)

# Save the model
model.save('model/simclr_preTrain_cifar10.h5')
print("Model saved as model/simclr_preTrain_cifar10.h5")

# Calculate accuracy on the training set
train_accuracy = model.evaluate(cifar_train_images, cifar_train_labels, verbose=0)[1]
print(f"Training Accuracy: {train_accuracy:.4f}")

# Function to extract features from training images
def extract_features(images, labels):
    features = model.predict(images)
    return features, labels

# Extract features from training set
train_features, train_labels = extract_features(cifar_train_images, cifar_train_labels)
