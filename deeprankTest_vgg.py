import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg16
import pickle
import time

torch.cuda.empty_cache()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
start_time = time.time()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the pre-trained VGG model with adjusted classifier
def load_model(save_path):
    model = vgg16(pretrained=True)
    model.to(device)
    # Modify classifier to match saved model
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Keep only up to penultimate layer
    
    # Load the state_dict with adjusted key names
    state_dict = torch.load(save_path)
    
    # Adjust keys if they have "base_model." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("base_model.", "") if k.startswith("base_model.") else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to allow minor mismatches
   
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
    
    with open('cosine_sim/cosine_sim_deeprank_vgg_flower.pkl', 'wb') as f:
        pickle.dump(similarities, f)
    
    print("Path Saved: cosine_sim/cosine_sim_deeprank_vgg_flower.pkl")



if __name__ == "__main__":
    with open('test/flower_image_pairs_test.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)
        
    calculate_accuracy("model/deeprank_vgg.pth", image_pairs_test, threshold=0.2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

