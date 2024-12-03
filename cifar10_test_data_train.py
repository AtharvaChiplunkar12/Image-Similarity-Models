import tensorflow as tf
import numpy as np
import pickle
import random

# Load CIFAR-10 dataset (only train data)
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

# Flatten the label array for easy indexing
y_train = y_train.flatten()

# Function to create image pairs
def create_image_pairs(images, labels, num_images_per_class):
    image_pairs = []
    num_classes = 10  # CIFAR-10 has 10 classes
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Create pairs
    for cls in range(num_classes):
        if len(class_indices[cls]) < num_images_per_class:
            continue  # Skip if not enough images in this class

        # Sample images for the current class
        sampled_indices = random.sample(list(class_indices[cls]), num_images_per_class)
        
        # Positive pairs (all combinations within the same class)
        for i in range(num_images_per_class):
            for j in range(i + 1, num_images_per_class):
                img1, img2 = images[sampled_indices[i]], images[sampled_indices[j]]
                image_pairs.append((img1, img2, 1))  # Positive pair with label 1

        # Negative pairs (pair each image with images from different classes)
        for other_cls in range(num_classes):
            if other_cls == cls:
                continue  # Skip the same class
            
            # Sample images from the other class
            sampled_other_indices = random.sample(list(class_indices[other_cls]), num_images_per_class)
            for i in range(num_images_per_class):
                img1 = images[sampled_indices[i]]
                img2 = images[sampled_other_indices[i]]
                image_pairs.append((img1, img2, 0))  # Negative pair with label 0

    return image_pairs

# Save pairs using pickle
def save_pairs(pairs, filename="cifar10_image_pairs_train.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(pairs, f)
    print(f"Pairs saved to {filename}")

# Create pairs
num_images_per_class = 100
all_image_pairs = create_image_pairs(x_train, y_train, num_images_per_class)

# Save the pairs
save_pairs(all_image_pairs)

print(f"Total pairs created: {len(all_image_pairs)}")
