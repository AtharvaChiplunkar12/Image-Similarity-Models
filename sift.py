import cv2
import numpy as np
from check import image_pairs_test

# Function to compute the similarity using SIFT
def compute_sift_similarity(image_pairs):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    results = []

    for img1_path, img2_path, ground_truth in image_pairs:
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # Use BFMatcher to find matches
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Calculate accuracy
        if len(good_matches) > 0:
            predicted = 1 if len(good_matches) > 10 else 0  
        else:
            predicted = 0
        
        results.append((img1_path, img2_path, ground_truth, predicted))

    return results


# Calculate the results
results = compute_sift_similarity(image_pairs_test)

# Calculate accuracy
correct_predictions = sum(1 for _, _, gt, pred in results if gt == pred)
accuracy = correct_predictions / len(image_pairs_test) * 100

# Print the results
for img1, img2, gt, pred in results:
    print(f"Comparing {img1} and {img2} - Ground Truth: {gt}, Predicted: {pred}")

print(f"Accuracy: {accuracy:.2f}%")
