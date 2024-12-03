import cv2
import numpy as np
import pickle

# Load the image pairs from the pickle file
with open('cifar10_image_pairs_train.pkl', 'rb') as f:
    image_pairs_test = pickle.load(f)

def load_image_pairs(image_pairs):
    images = []
    labels = []
    for img1, img2, label in image_pairs:
        # Append image pairs and labels
        images.append((img1, img2))
        labels.append(label)
    return images, labels

def compute_orb_matches(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create a BFMatcher object with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

def calculate_accuracy(image_pairs, labels):
    correct = 0
    total = len(labels)
    results = []

    for (img1, img2), label in zip(image_pairs, labels):
        matches = compute_orb_matches(img1, img2)
        
        # Define a threshold for a good match
        good_match_threshold = 20
        
        good_matches = [m for m in matches if m.distance < good_match_threshold]
        
        # Determine if the prediction is correct
        predicted_label = 1 if len(good_matches) > 0 else 0

        if predicted_label == label:
            correct += 1

        # Append the result with ground truth and predicted label
        results.append((label, predicted_label))
        
    accuracy = correct / total
    return accuracy, results

# Load image pairs and labels
image_pairs, labels = load_image_pairs(image_pairs_test)

# Calculate accuracy
accuracy, results = calculate_accuracy(image_pairs, labels)

# Print out results
for i, (gt, pred) in enumerate(results):
    print(f"Pair {i} - Ground Truth: {gt}, Predicted: {pred}")

print(f'Accuracy: {accuracy * 100:.2f}%')
