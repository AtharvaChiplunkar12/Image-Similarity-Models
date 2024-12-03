import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
import numpy as np
import pickle
import time

# Function to create base network with ResNet50 pre-trained on ImageNet without fine-tuning
def create_base_network(input_shape):
    # Load the pre-trained ResNet50 model (no top layers)
    base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)  # Using pre-trained weights from ImageNet
    base_model.trainable = False  # Freeze the pre-trained model (do not fine-tune)
    
    # Build the rest of the network
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu')  # Optional dense layer to increase complexity
    ])
    return model

# Function to compute cosine similarity using TensorFlow
def cosine_similarity_tf(x, y):
    # Compute the dot product
    dot_product = tf.reduce_sum(tf.multiply(x, y), axis=-1)
    # Compute the norms (magnitudes) of the vectors
    norm_x = tf.linalg.norm(x, axis=-1)
    norm_y = tf.linalg.norm(y, axis=-1)
    # Compute cosine similarity
    similarity = dot_product / (norm_x * norm_y)
    return similarity

# Create base network using pre-trained ResNet50 (without fine-tuning)
input_shape = (32, 32, 3)  # CIFAR-10 images have shape (32, 32, 3)
base_network = create_base_network(input_shape)

# Define the Siamese network
input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

# Pass each input through the base network
feat_vecs_a = base_network(input_a)
feat_vecs_b = base_network(input_b)

# Compute the cosine similarity between the feature vectors using the custom function
similarity = layers.Lambda(lambda tensors: cosine_similarity_tf(tensors[0], tensors[1]))([feat_vecs_a, feat_vecs_b])

# Reshape similarity output to match input shape of the Dense layer
similarity = layers.Reshape((1,))(similarity)

# Add a Dense layer for classification
output = layers.Dense(1, activation="sigmoid")(similarity)

siamese_network = Model(inputs=[input_a, input_b], outputs=output)

# Function to preprocess a single image
def preprocess_image(img):
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image to [0, 1]
    return img

# Load the image pairs from the pickle file
with open('cifar10_image_pairs_test.pkl', 'rb') as f:
    image_pairs_test = pickle.load(f)

# Function to predict similarity for the pairs in the test set
def predict_similarity_for_pairs(image_pairs_test):
    total = len(image_pairs_test)
    correct_predictions = 0
    print(f"Total test pairs: {total}")
    similarities = []
    
    for pair in image_pairs_test:
        img1, img2, actual_label = pair  # Extract the pair and the actual label
        
        # Preprocess the images
        img1_preprocessed = preprocess_image(img1)
        img2_preprocessed = preprocess_image(img2)
        
        # Predict similarity (cosine similarity)
        similarity_score = siamese_network.predict([img1_preprocessed, img2_preprocessed])[0][0]
        
        similarities.append((similarity_score, actual_label))
        print(f"Similarity: {similarity_score}, Label: {actual_label}")
    
    # Save the similarity scores to a file
    cosine_sim_array = np.array(similarities)
    save_path = 'cosine_sim/cosine_sim_siameseNet_cifar10.npy'
    np.save(save_path, cosine_sim_array)
    print(f"Cosine similarities saved to {save_path}")

# Start the testing process and calculate accuracy
start_time = time.time()
predict_similarity_for_pairs(image_pairs_test)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Testing completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
