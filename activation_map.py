"""
Activation Map Visualization for Eye Tracking Model

This script implements Grad-CAM to visualize which regions of input images 
the model is focusing on when predicting gaze coordinates.
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from model import Model
import pygame
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The trained model
            target_layer: The layer to use for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture activations and gradients
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register the hooks on the target layer
        handle1 = self.target_layer.register_forward_hook(forward_hook)
        handle2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.append(handle1)
        self.hooks.append(handle2)
        
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
            
    def generate_cam(self, input_image):
        """Generate the class activation map for the input image."""
        # Forward pass
        output = self.model(input_image)
        
        # Create a vector of ones for backpropagation
        # This is a simple approach for regression; we can backpropagate with respect to all outputs
        one_hot = torch.ones(output.size()).to(input_image.device)
        
        # Zero all existing gradients
        self.model.zero_grad()
        
        # Backward pass with the one-hot vector
        output.backward(gradient=one_hot)
        
        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of the gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of the activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU on the CAM
        cam = torch.relu(cam)
        
        # Normalize the CAM
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Resize CAM to the size of the input image
        cam = cam.squeeze().cpu().data.numpy()
        
        return cam

class ActivationMapVisualizer:
    def __init__(self, model_path="best_model.pth", width=1500, height=840):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Eye Tracker Activation Map")
        self.clock = pygame.time.Clock()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        self.model.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform - same as in training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize Grad-CAM with the target layer (last convolutional layer)
        self.grad_cam = GradCAM(self.model, self.model.pass_2[0])  # Use the first conv layer in pass_2
        
    def process_webcam_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Process frame the same way as in dataset generation
        # Crop to 1080x1080
        height, width = frame.shape[:2]
        center_x = width // 2
        crop_size = min(height, width)
        left_x = center_x - crop_size // 2
        right_x = center_x + crop_size // 2
        
        cropped_frame = frame[:, left_x:right_x, :]
        
        # Resize to 224x224 for the model
        model_input = cv2.resize(cropped_frame, (224, 224))
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Display image
        display_frame = cv2.resize(cropped_frame, (400, 400))  # Resize for display
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        return model_input, display_frame
    
    def predict_with_activation_map(self, image):
        # Convert image to tensor and normalize
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get activation map
        cam = self.grad_cam.generate_cam(image_tensor)
        
        # Resize CAM to the size of the input image
        cam = cv2.resize(cam, (224, 224))
        
        # Apply colormap to CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        original_image = np.array(image)
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        # Get prediction
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Get x, y coordinates (denormalize)
        x = predictions[0, 0].item() * self.width
        y = predictions[0, 1].item() * self.height
        
        return x, y, superimposed, cam
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        # Save current activation map
                        if hasattr(self, 'last_cam') and hasattr(self, 'last_superimposed'):
                            plt.figure(figsize=(12, 6))
                            plt.subplot(1, 2, 1)
                            plt.imshow(self.last_original_img)
                            plt.title('Original Image')
                            plt.subplot(1, 2, 2)
                            plt.imshow(self.last_superimposed)
                            plt.title('Activation Map')
                            plt.colorbar()
                            plt.tight_layout()
                            plt.savefig('activation_map.png')
                            plt.close()
                            print("Saved activation map to activation_map.png")
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Process webcam frame
            model_input, display_frame = self.process_webcam_frame()
            
            if model_input is not None and display_frame is not None:
                # Make prediction with activation map
                pred_x, pred_y, superimposed, cam = self.predict_with_activation_map(model_input)
                
                # Save for potential saving
                self.last_cam = cam
                self.last_superimposed = superimposed
                self.last_original_img = model_input
                
                # Convert numpy array to pygame surface for display
                webcam_surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
                
                # Convert activation map to pygame surface
                superimposed_resized = cv2.resize(superimposed, (400, 400))
                heatmap_surface = pygame.surfarray.make_surface(superimposed_resized.swapaxes(0, 1))
                
                # Display webcam feed in top-left corner
                self.screen.blit(webcam_surface, (20, 20))
                
                # Display heatmap next to it
                self.screen.blit(heatmap_surface, (440, 20))
                
                # Draw predicted gaze position
                pygame.draw.circle(self.screen, (0, 255, 0), (int(pred_x), int(pred_y)), 20)
                
                # Draw a line from the center to the predicted position
                pygame.draw.line(self.screen, (255, 0, 0), (self.width//2, self.height//2), (int(pred_x), int(pred_y)), 2)
                
                # Display prediction coordinates
                font = pygame.font.Font(None, 36)
                text = font.render(f"Predicted: ({pred_x:.1f}, {pred_y:.1f})", True, (255, 255, 255))
                self.screen.blit(text, (20, 430))
                
                # Display instructions
                instructions = font.render("Press 'S' to save activation map, ESC to exit", True, (255, 255, 255))
                self.screen.blit(instructions, (20, 480))
            
            pygame.display.flip()
            self.clock.tick(30)  # 30 fps
        
        # Clean up
        self.grad_cam.remove_hooks()
        self.cap.release()
        pygame.quit()

def analyze_dataset_samples(dataset_path, model_path="best_model.pth", num_samples=5):
    """
    Analyze and visualize activation maps for random samples from the dataset
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.load_model(model_path)
    model.to(device)
    model.eval()
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, model.pass_2[0])
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get list of image files
    import glob
    image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
    
    if not image_files:
        print(f"No images found in {dataset_path}")
        return
    
    # Select random samples
    import random
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, img_path in enumerate(selected_files):
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Transform image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Generate activation map
        cam = grad_cam.generate_cam(img_tensor)
        
        # Resize CAM to the size of the input image
        cam = cv2.resize(cam, (224, 224))
        
        # Apply colormap to CAM
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        superimposed = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
        
        # Get prediction
        with torch.no_grad():
            predictions = model(img_tensor)
        
        x = predictions[0, 0].item()
        y = predictions[0, 1].item()
        
        # Plot
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_array)
        plt.title(f"Original - Sample {i+1}")
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(cam, cmap='jet')
        plt.title(f"Activation Map - Sample {i+1}")
        plt.colorbar()
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(superimposed)
        plt.title(f"Overlay - Pred: ({x:.2f}, {y:.2f})")
    
    plt.tight_layout()
    plt.savefig('dataset_activation_maps.png')
    plt.close()
    print("Saved activation maps to dataset_activation_maps.png")
    
    # Clean up
    grad_cam.remove_hooks()

if __name__ == "__main__":
    # Run live activation map visualization
    visualizer = ActivationMapVisualizer()
    visualizer.run()
    
    # Uncomment to analyze dataset samples
    # analyze_dataset_samples("Dataset", num_samples=5) 