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
        
        # Set webcam resolution to match training data
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual webcam dimensions
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam resolution: {actual_width}x{actual_height}")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        self.model.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform for RGB input (EXACTLY as in training)
        class ToTensorRGB(object):
            """Convert numpy image to PyTorch tensor and normalize."""
            def __call__(self, image):
                # Convert from numpy to tensor and normalize
                return torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        self.transform = transforms.Compose([
            ToTensorRGB(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # For displaying webcam feed
        self.webcam_surface = None
    
    def process_webcam_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Process frame EXACTLY the same way as in dataset generation
        # Crop to 1080x1080 (HARDCODED like in training)
        height, width = frame.shape[:2]
        
        # Use EXACTLY the same crop coordinates as in training
        left_x = 1920 // 2 - 1080 // 2
        right_x = 1920 // 2 + 1080 // 2
        
        # Ensure we have enough pixels
        if width < right_x or height < 1080:
            print(f"Warning: Webcam resolution {width}x{height} is too small for 1080x1080 crop")
            return None, None
        
        # Crop to 1080x1080 (same as training)
        cropped_frame = frame[:, left_x:right_x, :]
        
        # Resize to 224x224 for the model (same as training)
        model_input = cv2.resize(cropped_frame, (224, 224))
        # Convert BGR to RGB EXACTLY as in training
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        
        # Display image (resize for display)
        display_frame = cv2.resize(cropped_frame, (400, 400))
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        return model_input, display_frame
    
    def predict_gaze(self, image):
        # Convert image to tensor and normalize
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Get normalized coordinates (0-1) for consistency with training
        norm_x = predictions[0, 0].item()
        norm_y = predictions[0, 1].item()
        
        print(f"Predicted normalized: x={norm_x:.3f}, y={norm_y:.3f}")
        
        # Get x, y coordinates (denormalize for screen display)
        # These coordinates are relative to the training screen size (1500x840)
        x = norm_x * self.width
        y = norm_y * self.height
        
        return x, y, norm_x, norm_y
    
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
                pred_x, pred_y, norm_x, norm_y = self.predict_gaze(model_input)
                
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
                text1 = font.render(f"Screen: ({pred_x:.1f}, {pred_y:.1f})", True, (255, 255, 255))
                text2 = font.render(f"Normalized: ({norm_x:.3f}, {norm_y:.3f})", True, (255, 255, 255))
                self.screen.blit(text1, (20, 430))
                self.screen.blit(text2, (20, 470))
                
                # Add debug info
                text3 = font.render(f"Training screen: 1500x840", True, (255, 255, 255))
                text4 = font.render(f"Test screen: {self.width}x{self.height}", True, (255, 255, 255))
                self.screen.blit(text3, (20, 510))
                self.screen.blit(text4, (20, 550))
            else:
                # Display error message if frame processing failed
                font = pygame.font.Font(None, 48)
                error_text = font.render("Failed to process webcam frame", True, (255, 0, 0))
                self.screen.blit(error_text, (self.width//2 - 200, self.height//2))
            
            pygame.display.flip()
            self.clock.tick(30)  # 30 fps
        
        # Clean up
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    tester = LiveTester()
    tester.run()
