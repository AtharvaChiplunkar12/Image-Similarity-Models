from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import pickle
import time

start_time = time.time()

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to get feature vector for an image
def get_feature_vector(model, image_path):
    img = load_and_preprocess_image(image_path)
    feature_vector = model.predict(img)
    return feature_vector

# Compare two images and return cosine similarity
def compare_images(model, image_path1, image_path2):
    feature_vector1 = get_feature_vector(model, image_path1)
    feature_vector2 = get_feature_vector(model, image_path2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vector1, feature_vector2)
    return similarity[0][0]  # Return the similarity score

# Function to compute accuracy
def calculate_accuracy(model, image_pairs, threshold=0.4):
    correct_predictions = 0
    total_pairs = len(image_pairs)
    
    similarities = []
    for image1, image2, label in image_pairs:
        if os.path.exists(image1) and os.path.exists(image2):
            similarity = compare_images(model, image1, image2)
            similarities.append((similarity, label))
            
        else:
            print(f"One or both image paths are invalid: {image1}, {image2}")
    
    with open('cosine_sim_simclr_flower.pkl', 'wb') as f:
        pickle.dump(similarities, f)
    print(f"Saved Path: cosine_sim_simclr_flower.pkl")

# Main execution block
if __name__ == "__main__":
    # Load the model
    model = load_model('model/simclr_preTrain_flower.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    print("Model loaded successfully.")


    with open('test/flower_image_pairs_test.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)
    
    # Calculate accuracy for the model
    calculate_accuracy(model, image_pairs_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
