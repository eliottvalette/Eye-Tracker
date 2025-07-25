"""
Training pipeline
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import cv2
import numpy as np
from model import Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

LOAD_MODEL = False
OVERFIT_FIRST_BATCH = True

class ToTensorRGB(object):
    """Convert numpy image to PyTorch tensor and normalize."""
    def __call__(self, image):
        # Convert from numpy to tensor and normalize
        return torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

# New grayscale tensor converter
class ToTensorGray(object):
    """Convert numpy RGB image to single-channel grayscale tensor and normalize."""
    def __call__(self, image):
        # Convert RGB numpy array to grayscale (H x W)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Add channel dimension and convert to tensor
        gray = np.expand_dims(gray, axis=2)  # H x W x 1
        return torch.from_numpy(gray.transpose((2, 0, 1))).float() / 255.0

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

    if OVERFIT_FIRST_BATCH:
        # Before Launching the real training, we'll overfit on one batch to set up the weights
        model.train()
        # Get 8 batches for overfitting
        train_iter = iter(train_loader)
        images_list = []
        targets_list = []
        for _ in range(2):
            batch_images, batch_targets = next(train_iter)
            images_list.append(batch_images)
            targets_list.append(batch_targets)
        
        # Concatenate all batches
        images = torch.cat(images_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        images = images.to(device)
        targets = targets.to(device)
        
        print("Overfitting on a single batch to set up weights...")
        # Train for a few iterations on this single batch
        for i in range(200):  # 200 iterations should be enough to overfit
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Overfit iteration {i+1}/100, Loss: {loss.item():.6f}")
                
                # If loss is very low, we can stop early
                if loss.item() < 0.001:
                    print("Achieved very low loss, stopping overfit phase.")
                    break
        
        print("Overfit phase complete. Starting main training loop.")
    
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
    
    return train_losses, val_losses


def visualize_predictions(model, loader, device, name):
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
        for images, targets in tqdm(loader, desc=f"Generating predictions for {name}"):
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
    
    # Calculate distances between predicted and actual coordinates
    distances = np.sqrt(np.sum((pred_coords - actual_coords)**2, axis=1))
    
    # Normalize distances for color mapping (0 to 1 range)
    max_dist = np.max(distances)
    if max_dist > 0:
        norm_distances = distances / max_dist
    else:
        norm_distances = distances
    
    # Create color map (red to green)
    colors = []
    for d in norm_distances:
        # Convert distance to RGB: green (0,1,0) for good predictions, red (1,0,0) for bad ones
        r = min(1.0, d * 2)        # More red as distance increases
        g = min(1.0, 2 - d * 2)    # Less green as distance increases
        b = 0.0                    # No blue component
        colors.append([r, g, b])
    
    # Plot the results
    plt.figure(figsize=(10, 10))
    
    # Plot actual coordinates with very low opacity
    plt.scatter(actual_coords[:, 0], actual_coords[:, 1], c='blue', alpha=0.2, label='Actual')
    
    # Plot predicted coordinates with color based on distance
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c=colors, alpha=0.7, label='Predicted')
    
    # Add reference lines for x=y
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Set limits (assuming coordinates are normalized between 0 and 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Create custom legend elements for color gradient
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Good prediction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Average prediction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Poor prediction'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.2, markersize=10, label='Actual position')
    ]
    
    plt.title('Gaze Estimation Results: Actual vs Predicted Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(handles=legend_elements)
    plt.grid(alpha=0.3)
    
    # Add text showing average distance
    avg_distance = np.mean(distances)
    plt.text(0.05, 0.95, f'Average Error: {avg_distance:.4f}', transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.savefig(f'prediction_visualization_{name}.png')
    print(f"Visualization saved as prediction_visualization_{name}.png")

def get_proper_train_val_split():
    """
    Create train/val split based on original images to prevent data leakage
    
    Returns:
        train_indices, val_indices: Indices for proper train/val split in the augmented dataset
    """
    # Load original dataset
    original_df = pd.read_csv("dataset.csv")
    
    # Load augmented dataset
    augmented_df = pd.read_csv("augmented_dataset.csv")
    
    # Get original image filenames (without path)
    original_filenames = [os.path.basename(img_path).split('.')[0] for img_path in original_df['img_filename']]
    
    # Split original filenames into train/val sets
    train_size = int(0.9 * len(original_filenames))
    indices = np.arange(len(original_filenames))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_filenames = [original_filenames[i] for i in train_indices]
    val_filenames = [original_filenames[i] for i in val_indices]
    
    # Find corresponding augmented indices
    train_aug_indices = []
    val_aug_indices = []
    
    for idx, row in augmented_df.iterrows():
        # Extract original filename from augmented filename
        # Example: "Augmented_Dataset/1751743551_1.jpg" -> "1751743551"
        original_name = os.path.basename(row['img_filename']).split('_')[0]
        
        if original_name in train_filenames:
            train_aug_indices.append(idx)
        elif original_name in val_filenames:
            val_aug_indices.append(idx)
    
    print(f"Original dataset: {len(original_df)} images")
    print(f"Train set: {len(train_filenames)} original images, {len(train_aug_indices)} augmented images")
    print(f"Val set: {len(val_filenames)} original images, {len(val_aug_indices)} augmented images")
    
    return train_aug_indices, val_aug_indices

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Define transforms for grayscale images
    transform = transforms.Compose([
        ToTensorGray(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset
    full_dataset = EyeTrackerDataset(csv_file="augmented_dataset.csv", transform=transform)
    
    # Check if dataset is not empty
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Please run dataset_generator.py first.")
    
    # Get proper train/val split indices to prevent data leakage
    train_indices, val_indices = get_proper_train_val_split()
    
    # Create train and validation datasets using the indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Print dataset size
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders - set num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    model = Model()
    if LOAD_MODEL:
        print("Loading best model...")
        model.load_model("best_model.pth")
    
    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # Train the model with early stopping
        num_epochs = 50
        train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=15)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Loading best model for visualization...")
    
    # Load the best model for visualization
    model = Model()
    model.load_model("best_model.pth")
    model.to(device)
    
    # Visualize predictions
    visualize_predictions(model, train_loader, device, "train")
    visualize_predictions(model, val_loader, device, "val")
    
    print("Training and visualization complete!")

if __name__ == "__main__":
    main()