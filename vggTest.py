import torch
import time
from torchvision import transforms
from torchvision import models
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the trained model
torch.cuda.empty_cache()
start_time = time.time()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 5)  # Adjust for your number of classes
model.load_state_dict(torch.load('model/vgg_finetuned_flowers.pth'))
model.to(device)
model.eval()

# Define image preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = data_transforms(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy().flatten()  # Move to CPU and flatten

# Calculate accuracy based on cosine similarity and labels
def evaluate_image_pairs(image_pairs):
    correct_predictions = 0
    similarities = []
    for img1, img2, label in image_pairs:
        features1 = extract_features(img1)
        features2 = extract_features(img2)
        
        similarity = cosine_similarity([features1], [features2])[0][0]
        similarities.append((similarity, label))
        print("cosine_similarity: " + str(similarity) + " label: " + str(label))

    with open('cosine_sim_vgg_flower.pkl', 'wb') as f:
        pickle.dump(similarities, f)
    print(f"Saved Path: cosine_sim_vgg_flower.pkl")



with open('test/flower_image_pairs_test.pkl', 'rb') as f:
        image_pairs_test = pickle.load(f)
        
# Evaluate accuracy on your test pairs
evaluate_image_pairs(image_pairs_test)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")