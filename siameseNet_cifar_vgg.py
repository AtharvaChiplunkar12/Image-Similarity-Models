import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16  # Replaced ResNet50 with VGG16
import numpy as np
import random

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()  # Only use training data

# Normalize the images
x_train = x_train / 255.0

# Function to create pairs of images and labels (1 if same class, 0 if different class)
def make_pairs(images, labels):
    pairs, pair_labels = [], []
    num_classes = 10  # CIFAR-10 has 10 classes
    class_idx = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for idx, label in enumerate(labels):
        pos_idx = random.choice(class_idx[label[0]])
        pairs += [[images[idx], images[pos_idx]]]
        pair_labels += [1]
        
        neg_label = (label[0] + random.randint(1, num_classes - 1)) % num_classes
        neg_idx = random.choice(class_idx[neg_label])
        pairs += [[images[idx], images[neg_idx]]]
        pair_labels += [0]
        
    return np.array(pairs), np.array(pair_labels)

# Create pairs for the training set
train_pairs, train_labels = make_pairs(x_train, y_train)

# Define VGG16 as the base model for feature extraction
def create_base_network(input_shape):
    base_model = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
    base_model.trainable = False  # Freeze the base model
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu')
    ])
    return model

input_shape = (32, 32, 3)  # CIFAR-10 images have shape (32, 32, 3)
base_network = create_base_network(input_shape)

# Define the Siamese network
input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

# Pass each input through the base network
feat_vecs_a = base_network(input_a)
feat_vecs_b = base_network(input_b)

# Compute the L1 distance between the feature vectors
distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feat_vecs_a, feat_vecs_b])
output = layers.Dense(1, activation="sigmoid")(distance)
siamese_network = Model(inputs=[input_a, input_b], outputs=output)

# Define the contrastive loss function
def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Compile the model
siamese_network.compile(optimizer=Adam(0.0001), loss=contrastive_loss, metrics=['accuracy'])

# Train the model
history = siamese_network.fit(
    [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
    batch_size=32,
    epochs=10
)

# Save the model
siamese_network.save('siamese_network_vgg_cifar10.h5')
print("Model saved to siamese_network_vgg_cifar10.h5")
