import os
import pickle
import random
from itertools import combinations, product

# Path to the dataset folder with subfolders for each class
dataset_path = "flowers"

# Number of images to select from each class
num_images_per_class = 20

# Dictionary to hold the image paths for each class
class_images = {}

# Step 1: Load up to 20 images from each class
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        # Get a random sample of 20 images for this class
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) > num_images_per_class:
            images = random.sample(images, num_images_per_class)
        class_images[class_folder] = images

# Step 2: Create pairs within the same class and label them as 1
image_pairs_test = []

for class_name, images in class_images.items():
    # Generate all unique pairs within the class
    same_class_pairs = combinations(images, 2)
    for img1, img2 in same_class_pairs:
        image_pairs_test.append((img1, img2, 1))

# Step 3: Create pairs between different classes and label them as 0
class_names = list(class_images.keys())

# Generate pairs of classes to avoid redundant pairs
for i in range(len(class_names)):
    for j in range(i + 1, len(class_names)):
        class1, class2 = class_names[i], class_names[j]
        
        # Use cartesian product to pair each image from class1 with each image from class2
        diff_class_pairs = product(class_images[class1], class_images[class2])
        for img1, img2 in diff_class_pairs:
            image_pairs_test.append((img1, img2, 0))

# Step 4: Shuffle the list to mix same-class and different-class pairs
random.shuffle(image_pairs_test)
print(f"Total Pairs: {len(image_pairs_test)}")

with open('test/flower_image_pairs_test.pkl', 'wb') as f:
    pickle.dump(image_pairs_test, f)
    
