import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize
import tensorflow as tf
import pickle
import os
import time
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input

# Preprocess the image to match ResNet50 input requirements
def preprocess_image(img_array):
    img_array = np.array(img_array, dtype=np.float32)  # Ensure float32 type
    img_array = tf.convert_to_tensor(img_array)  # Convert to TensorFlow tensor
    img_array = tf.image.resize(img_array, (224, 224))  # Resize to (224, 224)
    img_array = preprocess_input(img_array)  # Standard ResNet preprocessing
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Extract feature vector for an image
def get_feature_vector(model, img_array):
    img = preprocess_image(img_array)  # Preprocess the image
    feature_vector = model.predict(img, verbose=0)  # Predict and extract features
    return feature_vector

# Compare two images and return cosine similarity
def compare_images(model, img_array1, img_array2):
    feature_vector1 = get_feature_vector(model, img_array1)  # Extract feature vector 1
    feature_vector2 = get_feature_vector(model, img_array2)  # Extract feature vector 2

    # Normalize feature vectors
    feature_vector1 = normalize(feature_vector1, axis=1)
    feature_vector2 = normalize(feature_vector2, axis=1)

    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vector1, feature_vector2)
    return similarity[0][0]  # Return scalar similarity value

# Compute accuracy and save cosine similarities
def calculate_accuracy(model, image_pairs, threshold=0.5, save_path='cosine_sim/cosine_sim_resnet_cifar10.npy'):
    cosine_sim = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists

    for img_array1, img_array2, label in image_pairs:
        similarity = compare_images(model, img_array1, img_array2)  # Compute similarity
        cosine_sim.append((similarity, label))  # Store similarity and label
        print(f"Similarity: {similarity:.4f}, Label: {label}")

    # Save cosine similarity results
    np.save(save_path, np.array(cosine_sim))
    print(f"Cosine similarities saved to {save_path}")

# Main execution block
if __name__ == "__main__":
    start_time = time.time()

    # Load the ResNet50 feature extractor
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor_model = Model(
        inputs=base_model.input,
        outputs=GlobalAveragePooling2D()(base_model.output)  # Add global pooling layer
    )
    print("Feature extractor model loaded successfully.")

    # Load test image pairs from a pickle file
    with open('cifar10_image_pairs_test.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)

    # Calculate accuracy for the model
    calculate_accuracy(feature_extractor_model, image_pairs_test)

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
