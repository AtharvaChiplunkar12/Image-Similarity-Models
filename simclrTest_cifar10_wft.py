import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import time

# Load the pre-trained SimCLR model without fine-tuning
hub_path = 'gs://simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/hub/'
model = hub.KerasLayer(hub_path, input_shape=(224, 224, 3), trainable=False)

# Function to preprocess an image
def preprocess_image(img_array):
    img_array = np.array(img_array) / 255.0  # Normalize the image
    img_array = tf.image.resize(img_array, (224, 224))  # Resize to match model input
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to get feature vector for an image
def get_feature_vector(model, img_array):
    img = preprocess_image(img_array)
    feature_vector = model(img)  # Get the feature vector from the model
    return feature_vector.numpy()  # Ensure it's a NumPy array for processing

# Compare two images and return cosine similarity
def compare_images(model, img_array1, img_array2):
    feature_vector1 = get_feature_vector(model, img_array1)
    feature_vector2 = get_feature_vector(model, img_array2)
    
    # Normalize feature vectors
    feature_vector1 = normalize(feature_vector1)
    feature_vector2 = normalize(feature_vector2)

    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vector1, feature_vector2)
    return similarity[0][0]  # Return the similarity score

# Function to compute accuracy
def calculate_accuracy(model, image_pairs, save_path='cosine_sim/cosine_sim_simclr_cifar10.npy'):
    cosine_sim = []
    for img_array1, img_array2, label in image_pairs:
        similarity = compare_images(model, img_array1, img_array2)
        cosine_sim.append((similarity, label))
        print(f"Similarity: {similarity}, Label: {label}")
    
    # Convert the list to a NumPy array
    cosine_sim_array = np.array(cosine_sim)

    # Save the array to a file
    np.save(save_path, cosine_sim_array)
    print(f"Cosine similarities saved to {save_path}")

# Main execution block
if __name__ == "__main__":
    start_time = time.time()
    
    # Load image pairs for testing (use the correct path if it's for testing or training)
    with open('cifar10_image_pairs_train.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)
        
    # Calculate accuracy for the model
    calculate_accuracy(model, image_pairs_test)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
