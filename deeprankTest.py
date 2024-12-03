import pickle
import torch
import torch.nn as nn
from PIL import Image  
import torchvision.transforms as transforms
from resnetModel import ResNetModel

import time
torch.cuda.empty_cache()
start_time = time.time()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the pre-trained model
def load_model(save_path):
    model = ResNetModel()  
    model.load_state_dict(torch.load(save_path))  
    model.to(device) 
    model.eval()  
   
    return model

# Compute similarity between two images
def compute_similarity(image1, image2, model):
    model.eval()
    with torch.no_grad():
        output1 = model(image1.unsqueeze(0).to(device))
        output2 = model(image2.unsqueeze(0).to(device))

        euclidean_distance = torch.norm(output1 - output2).item()
        cosine_similarity = nn.functional.cosine_similarity(output1, output2).item()

    return euclidean_distance, cosine_similarity

# Load and preprocess images
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

# Calculate accuracy
def calculate_accuracy(model_path, image_pairs, threshold):
    model = load_model(model_path)
    
    correct_predictions = 0
    total_pairs = len(image_pairs)
    similarities = []
    for img1_path, img2_path, actual_label in image_pairs:
        image1 = load_and_preprocess_image(img1_path)
        image2 = load_and_preprocess_image(img2_path)

        _, cosine_similarity = compute_similarity(image1, image2, model)
        similarities.append((cosine_similarity, actual_label))
        print("cosine_similarity: " + str(cosine_similarity) + " label: " + str(actual_label))
        
    with open('cosine_sim/cosine_sim_deeprank_flower.pkl', 'wb') as f:
        pickle.dump(similarities, f)
    print("Saved Path: cosine_sim/cosine_sim_deeprank_flower.pkl")
if __name__ == "__main__":
    with open('test/flower_image_pairs_test.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)
    
    calculate_accuracy("model/deeprank.pth", image_pairs_test, threshold=0.4)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
