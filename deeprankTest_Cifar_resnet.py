import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image
import pickle
import os
import torchvision.models as models
import torch.nn as nn
import time

# Define image transformations globally
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet input
    transforms.ToTensor(),          # Convert image to tensor
])

# Function to preprocess the image
def preprocess_image(img):
    # If the input is a NumPy array, convert it to a PIL image
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8))  # Convert normalized numpy array back to PIL Image

    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0).to(device)  # Add batch dimension and send to device
    return img

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResNet-based model for feature extraction
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        return self.resnet(x)

# Load the model and adapt it to output feature vectors
def load_feature_extractor(model_path):
    model = ResNetModel()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # strict=False to ignore the mismatch in final layer
    model = model.to(device)
    model.eval()
    return model

# Function to get feature vector for an image
def get_feature_vector(model, img):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        feature_vector = model(img_tensor)
    return feature_vector.cpu().numpy()

# Compare two images and return cosine similarity
def compare_images(model, img1, img2):
    feature_vector1 = get_feature_vector(model, img1)
    feature_vector2 = get_feature_vector(model, img2)

    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vector1, feature_vector2)
    return similarity[0][0]  # Return the similarity score

# Function to compute similarity and save results
def calculate_similarity_and_save(model, image_pairs, save_path='cosine_sim/cosine_sim_deeprank_cifar10.npy'):
    cosine_sim = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for img1, img2, label in image_pairs:
        similarity = compare_images(model, img1, img2)
        cosine_sim.append((similarity, label))
        print(f"Similarity: {similarity}, Label: {label}")
    
    # Convert to numpy array and save
    cosine_sim_array = np.array(cosine_sim, dtype=object)  # dtype=object to store tuples
    np.save(save_path, cosine_sim_array)
    print(f"Cosine similarities saved to {save_path}")

# Main execution block
if __name__ == "__main__":
    start_time = time.time()
    model_path = "model/cifar10_deeprank.pth"
    
    # Load the feature extractor model
    feature_extractor_model = load_feature_extractor(model_path)
    print("Feature extractor model loaded successfully.")

    # Load the image pairs
    with open('cifar10_image_pairs_train.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)
        
    # Calculate cosine similarity and save results
    calculate_similarity_and_save(feature_extractor_model, image_pairs_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
