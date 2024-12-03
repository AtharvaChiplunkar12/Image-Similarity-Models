import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle
from torchvision import models

torch.cuda.empty_cache()
start_time = time.time()

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Load the saved model weights
model.load_state_dict(torch.load('resnet_finetuned_flowers.pth'))
model.eval()  # Set the model to evaluation mode

# Remove the last (classification) layer to extract features
feature_model = nn.Sequential(*list(model.children())[:-1]) 

# Set device (GPU if available)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
feature_model = feature_model.to(device)

# Define image preprocessing similar to training
def preprocess_image(img_path, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img

# Calculate cosine similarity
def get_cosine_similarity(feature1, feature2):
    # Compute cosine similarity between two feature vectors
    return cosine_similarity(feature1, feature2)[0][0]

def compare_image_pairs(model, image_pairs):
    correct = 0
    total = len(image_pairs)
    similarities = []
    for img1_path, img2_path, label in image_pairs:
        # Preprocess both images
        img1 = preprocess_image(img1_path).to(device)
        img2 = preprocess_image(img2_path).to(device)
        
        # Extract features
        with torch.no_grad():
            img1_features = model(img1).cpu().numpy().reshape(1, -1)  # Flatten to a 1D vector
            img2_features = model(img2).cpu().numpy().reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = get_cosine_similarity(img1_features, img2_features)
        
        similarities.append((similarity, label))
        print("cosine_similarity: " + str(similarity) + " label: " + str(label))
    
    with open('cosine_sim_resnet_flower.pkl', 'wb') as f:
        pickle.dump(similarities, f)
        
    print(f"Saved Path: cosine_sim_resnet_flower.pkl")

        


with open('test/flower_image_pairs_test.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)
        
# Run the comparison on the test pairs
compare_image_pairs(feature_model, image_pairs_test)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
