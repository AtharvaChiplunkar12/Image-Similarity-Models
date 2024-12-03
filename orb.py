import cv2
import numpy as np
from check import image_pairs_test

def load_image_pairs(image_pairs):
    images = []
    paths = []
    for image_path1, image_path2, _ in image_pairs:
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        images.append((img1, img2))
        paths.append((image_path1, image_path2))
    return images, paths

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

def calculate_accuracy(image_pairs, image_paths, image_pairs_test):
    correct = 0
    total = len(image_pairs_test)
    results = []

    for (img1, img2), (path1, path2), (_, _, label) in zip(image_pairs, image_paths, image_pairs_test):
        matches = compute_orb_matches(img1, img2)
        
        # Define a threshold for a good match
        good_match_threshold = 45  
        
        good_matches = [m for m in matches if m.distance < good_match_threshold]
        
        # If the number of good matches indicates similarity
        if len(good_matches) > 0:
            predicted_label = 1
        else:
            predicted_label = 0

        # Count correct predictions
        if predicted_label == label:
            correct += 1
        
        # Append the result with image paths, ground truth, and predicted label
        results.append((path1, path2, label, predicted_label))
        
    accuracy = correct / total
    return accuracy, results


# Load image pairs and their paths
image_pairs, image_paths = load_image_pairs(image_pairs_test)

# Calculate accuracy
accuracy, results = calculate_accuracy(image_pairs, image_paths, image_pairs_test)

# Print out results
for path1, path2, gt, pred in results:
    print(f"Comparing {path1} and {path2} - Ground Truth: {gt}, Predicted: {pred}")

print(f'Accuracy: {accuracy * 100:.2f}%')
