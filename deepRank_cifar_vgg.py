import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets
from torchvision.models import vgg16
import torch.nn.functional as F

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match VGG input size
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Custom Dataset for Triplet Loss
class TripletImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = dataset.targets  # Targets are available in CIFAR-10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_img, anchor_label = self.dataset[idx]
        positive_idx = random.choice([i for i in range(len(self.dataset)) if self.labels[i] == anchor_label and i != idx])
        positive_img, _ = self.dataset[positive_idx]
        negative_idx = random.choice([i for i in range(len(self.dataset)) if self.labels[i] != anchor_label])
        negative_img, _ = self.dataset[negative_idx]
        return anchor_img, positive_img, negative_img

# Create Triplet Dataset
triplet_dataset = TripletImageDataset(cifar10)
train_size = int(0.8 * len(triplet_dataset))
val_size = len(triplet_dataset) - train_size
train_dataset, val_dataset = random_split(triplet_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize VGG model, modified for our use
class VGGModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(VGGModel, self).__init__()
        self.base_model = vgg16(pretrained=True)
        # Retain feature extractor only, remove the final classification layer
        self.base_model.classifier = nn.Sequential(*list(self.base_model.classifier.children())[:-1])
        self.embedding_layer = nn.Linear(4096, embedding_dim)  # Embedding layer

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings

model = VGGModel().to(device)
criterion = nn.TripletMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Lower learning rate
scaler = GradScaler()

# Gradient clipping function
def clip_gradients(model, max_norm=1.0):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.nn.utils.clip_grad_norm_(param, max_norm)

# Function to calculate accuracy
def calculate_similarity_accuracy(anchor_output, positive_output, negative_output):
    pos_dist = F.pairwise_distance(anchor_output, positive_output)
    neg_dist = F.pairwise_distance(anchor_output, negative_output)
    return (pos_dist < neg_dist).float().mean().item()

# Training and validation function
def train_model(num_epochs, save_path):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()

            with autocast(enabled=torch.cuda.is_available()):
                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)
                loss = criterion(anchor_output, positive_output, negative_output)

            scaler.scale(loss).backward()
            clip_gradients(model)  # Clip gradients
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_accuracy += calculate_similarity_accuracy(anchor_output, positive_output, negative_output)

        avg_loss = running_loss / len(train_loader)
        avg_train_acc = train_accuracy / len(train_loader)

        model.eval()
        val_accuracy = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)

                val_accuracy += calculate_similarity_accuracy(anchor_output, positive_output, negative_output)

        avg_val_acc = val_accuracy / len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Train Similarity Accuracy: {avg_train_acc:.4f}, Validation Similarity Accuracy: {avg_val_acc:.4f}")

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

model_path = "model/deeprank_vgg_cifar10.pth"
num_epochs = 10
train_model(num_epochs, model_path)
