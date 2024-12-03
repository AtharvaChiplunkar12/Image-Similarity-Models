import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import numpy as np
import random

# Set image size and batch size
img_size = (224, 224)
batch_size = 32

# Load the flower dataset from the directory (adjust the path as necessary)
data_dir = "flowers"  # Replace with the path to your flower dataset

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode="int",
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=42
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode="int",
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Normalize images
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

# Function to create pairs of images and labels (1 if same class, 0 if different class)
def make_pairs(dataset, num_classes=5):
    images, labels = [], []
    for img, lbl in dataset:
        images.extend(img.numpy())
        labels.extend(lbl.numpy())

    images, labels = np.array(images), np.array(labels)
    pairs, pair_labels = [], []
    class_idx = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for idx, label in enumerate(labels):
        pos_idx = random.choice(class_idx[label])
        pairs += [[images[idx], images[pos_idx]]]
        pair_labels += [1]
        
        neg_label = (label + random.randint(1, num_classes - 1)) % num_classes
        neg_idx = random.choice(class_idx[neg_label])
        pairs += [[images[idx], images[neg_idx]]]
        pair_labels += [0]
        
    return np.array(pairs), np.array(pair_labels)

# Create pairs for the training dataset
train_pairs, train_labels = make_pairs(train_dataset)

# Define ResNet50 as the base model for feature extraction
def create_base_network(input_shape):
    base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
    base_model.trainable = False  # Freeze the base model
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu')
    ])
    return model

input_shape = img_size + (3,)  # Flower images resized to (224, 224, 3)
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

# Train the model using only training data
history = siamese_network.fit(
    [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
    batch_size=32,
    epochs=10
)

# Save the model
siamese_network.save('siamese_network_resnet_flowers.h5')
print("Model saved to siamese_network_resnet_flowers.h5")
