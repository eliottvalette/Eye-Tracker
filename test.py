"""
Real-time eye tracking test

This script loads the trained model and tests it in real-time using the webcam.
It displays the webcam feed and draws a circle at the predicted gaze position.
"""
import pygame
import numpy as np
import cv2
import torch
from model import Model
from torchvision import transforms

class LiveTester:
    def __init__(self, model_path="best_model.pth", width=1500, height=840):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Eye Tracker Live Test")
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
        
        # Define transform for grayscale input (consistent with training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # For displaying webcam feed
        self.webcam_surface = None
    
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
    
    def predict_gaze(self, image):
        # Convert image to tensor and normalize
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        print(f"Predicted x: {predictions[0, 0].item()}, y: {predictions[0, 1].item()}")
        
        # Get x, y coordinates (denormalize)
        x = predictions[0, 0].item() * self.width
        y = predictions[0, 1].item() * self.height
        
        return x, y
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Process webcam frame
            model_input, display_frame = self.process_webcam_frame()
            
            if model_input is not None and display_frame is not None:
                # Make prediction
                pred_x, pred_y = self.predict_gaze(model_input)
                
                # Convert numpy array to pygame surface for display
                webcam_surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
                
                # Display webcam feed in top-left corner
                self.screen.blit(webcam_surface, (20, 20))
                
                # Draw predicted gaze position
                pygame.draw.circle(self.screen, (0, 255, 0), (int(pred_x), int(pred_y)), 20)
                
                # Draw a line from the webcam to the predicted position
                pygame.draw.line(self.screen, (255, 0, 0), (self.width//2, self.height//2), (int(pred_x), int(pred_y)), 2)
                
                # Display prediction coordinates
                font = pygame.font.Font(None, 36)
                text = font.render(f"Predicted: ({pred_x:.1f}, {pred_y:.1f})", True, (255, 255, 255))
                self.screen.blit(text, (20, 430))
            
            pygame.display.flip()
            self.clock.tick(30)  # 30 fps
        
        # Clean up
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    tester = LiveTester()
    tester.run()
