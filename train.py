"""
Training pipeline
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import cv2
import numpy as np
from model import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms


class ToTensorRGB(object):
    """Convert numpy image to PyTorch tensor and normalize."""
    def __call__(self, image):
        # Convert from numpy to tensor and normalize
        return torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0


class EyeTrackerDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get coordinates (normalized)
        x = self.data.iloc[idx, 1]
        y = self.data.iloc[idx, 2]
        coordinates = torch.tensor([x, y], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, coordinates


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path="best_model.pth", patience=5):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Early stopping variables
    counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        val_progress_bar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for images, targets in val_progress_bar:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, targets)
                
                running_val_loss += val_loss.item() * images.size(0)
                val_progress_bar.set_postfix({'val_loss': val_loss.item()})
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save_model(save_path)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()
    
    return train_losses, val_losses


def visualize_predictions(model, val_loader, device):
    """
    Visualize the predicted vs actual gaze points on a 2D plane.
    
    Args:
        model: The trained model
        val_loader: Validation data loader
        device: Device to run inference on
    """
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store actual and predicted coordinates
    actual_coords = []
    pred_coords = []
    
    # Get predictions from validation set
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Generating predictions"):
            images = images.to(device)
            outputs = model(images)
            
            # Move to CPU and convert to numpy
            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()
            
            # Add to lists
            actual_coords.extend(targets)
            pred_coords.extend(outputs)
    
    # Convert to numpy arrays
    actual_coords = np.array(actual_coords)
    pred_coords = np.array(pred_coords)
    
    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.scatter(actual_coords[:, 0], actual_coords[:, 1], c='blue', alpha=0.5, label='Actual')
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', alpha=0.5, label='Predicted')
    
    # Add reference lines for x=y
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Set limits (assuming coordinates are normalized between 0 and 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.title('Gaze Estimation Results: Actual vs Predicted Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('prediction_visualization.png')
    plt.show()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Define transforms - use a proper class instead of lambda
    transform = transforms.Compose([
        ToTensorRGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = EyeTrackerDataset(csv_file="dataset.csv", transform=transform)
    
    # Check if dataset is not empty
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please run dataset_generator.py first.")
    
    # Split dataset into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders - set num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Initialize model
    model = Model()
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model with early stopping
    num_epochs = 50
    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=5)
    
    # Load the best model for visualization
    model = Model()
    model.load_model("best_model.pth")
    model.to(device)
    
    # Visualize predictions
    visualize_predictions(model, val_loader, device)
    
    print("Training and visualization complete!")


if __name__ == "__main__":
    main()