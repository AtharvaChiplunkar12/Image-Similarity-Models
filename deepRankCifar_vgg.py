import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from torchvision.models import vgg16
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

torch.cuda.empty_cache()
# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing for VGG input
    transforms.ToTensor(),
])

# Custom Dataset for CIFAR10 Triplet Loss
class CIFAR10TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [sample[1] for sample in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_img, anchor_label = self.dataset[idx]

        # Choose a positive sample (same class)
        positive_idx = random.choice([i for i in range(len(self.dataset)) if self.labels[i] == anchor_label and i != idx])
        positive_img, _ = self.dataset[positive_idx]

        # Choose a negative sample (different class)
        negative_idx = random.choice([i for i in range(len(self.dataset)) if self.labels[i] != anchor_label])
        negative_img, _ = self.dataset[negative_idx]

        return transform(anchor_img), transform(positive_img), transform(negative_img)

# Load CIFAR-10 dataset
cifar10_data = CIFAR10(root='data', train=True, download=True)
dataset = CIFAR10TripletDataset(cifar10_data)

# Split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Define the VGG model with a custom embedding layer
class VGGModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(VGGModel, self).__init__()
        # Load pretrained VGG16 model
        self.vgg = vgg16(pretrained=True)
        # Remove the classification layer
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        # Add an embedding layer
        self.embedding_layer = nn.Linear(4096, embedding_dim)

    def forward(self, x):
        x = self.vgg(x)
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize the embeddings

# Initialize model, loss function, and optimizer
model = VGGModel().to(device)
criterion = nn.TripletMarginLoss(margin=1.0)  # Adjust margin if necessary
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Reduced learning rate

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

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

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_accuracy += calculate_similarity_accuracy(anchor_output, positive_output, negative_output)

        # Calculate average loss and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        avg_train_acc = train_accuracy / len(train_loader)

        # Validation step
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

        # Optionally clear cache to manage memory
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Function to calculate accuracy for triplet loss
def calculate_similarity_accuracy(anchor_output, positive_output, negative_output):
    # Distance between anchor and positive
    pos_dist = F.pairwise_distance(anchor_output, positive_output)
    # Distance between anchor and negative
    neg_dist = F.pairwise_distance(anchor_output, negative_output)
    # Check if anchor is closer to positive than negative
    correct = (pos_dist < neg_dist).float()
    return correct.sum().item() / len(correct)

# Train the model
train_model(num_epochs=5, save_path="vgg_triplet_model.pth")
