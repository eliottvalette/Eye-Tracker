"""
Dataset generator

This script is used to generate a dataset of images and their corresponding x, y coordinates.
It uses a webcam to capture images and a moving circle to indicate the target on the screen.
The user has to look at the circle as it moves around the screen.

The dataset is saved in a csv file.
The csv file contains the following columns:
- img_filename: the filename of the image
- x: the x coordinate of the target
- y: the y coordinate of the target
"""
import pygame
import random as rd
import os
import pandas as pd
import cv2
import time
import threading

TIME = 120
CAPTURE_INTERVAL = 0.25  # Capture every 250ms

# Creat directory if it doesn't exist
os.makedirs("Dataset", exist_ok=True)

class DatasetGenerator:
    def __init__(self, width=1500, height=840):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dataset Generator")
        self.clock = pygame.time.Clock()
        self.temp_df = pd.DataFrame(columns=["img_filename", "x", "y"])
        
        # Initialize webcam once
        self.cap = cv2.VideoCapture(0)
        
        # Circle movement properties
        self.circle_x = width // 2
        self.circle_y = height // 2
        self.circle_vx = rd.choice([-1, 1]) * rd.uniform(3, 4)
        self.circle_vy = rd.choice([-1, 1]) * rd.uniform(3, 4)

        # Curve direction
        self.curve_x = rd.uniform(0.9, 1.1)
        self.curve_y = rd.uniform(0.9, 1.1)

        # Threading for continuous capture
        self.capture_running = False
        self.capture_thread = None
        
    def run_dataset_generator(self):
        start_time = time.time()
        
        # Start continuous capture in a separate thread
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self.continuous_capture)
        self.capture_thread.start()
        
        while time.time() - start_time < TIME:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.capture_running = False
                    return
            
            # Update circle position with advanced movement
            self.update_circle_position_advanced()
            
            # Fill the screen with black
            self.screen.fill((0, 0, 0))
            
            # Draw text on the screen
            font = pygame.font.Font(None, 96)
            time_remaining = TIME - (time.time() - start_time)
            text = font.render(f"{time_remaining:.1f}s", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(text, text_rect)

            # Add progress bar below the text
            progress_bar = pygame.Rect(0, self.height - 50, self.width, 20)
            pygame.draw.rect(self.screen, (255, 255, 255), progress_bar)
            pygame.draw.rect(self.screen, (0, 255, 0), (0, self.height - 50, self.width * (time.time() - start_time) / TIME, 20))

            # Draw the moving circle
            pygame.draw.circle(self.screen, (255, 0, 0), (int(self.circle_x), int(self.circle_y)), 20)
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        # Stop continuous capture
        self.capture_running = False
        if self.capture_thread:
            self.capture_thread.join()
        
        # Release webcam when done
        self.release_webcam()

    def update_circle_position_advanced(self):
        """Update circle position with advanced movement patterns"""
        # Calculate distance from center
        center_x = self.width // 2
        center_y = self.height // 2
        distance_from_center = ((self.circle_x - center_x)**2 + (self.circle_y - center_y)**2)**0.5
        
        # Maximum distance from center (diagonal)
        max_distance = ((self.width // 2)**2 + (self.height // 2)**2)**0.5
        
        # Speed multiplier: closer to center = faster (inverse relationship)
        # Normalize distance (0 at center, 1 at edge)
        normalized_distance = distance_from_center / max_distance
        
        # Speed multiplier: 2.0 at center, 1.0 at edge
        speed_multiplier = 2.0 - normalized_distance
        
        # Apply speed multiplier to velocities
        self.circle_x += self.circle_vx * self.curve_x * speed_multiplier
        self.circle_y += self.circle_vy * self.curve_y * speed_multiplier
        
        # Check if circle is hitting the edges
        if self.circle_x <= 20 or self.circle_x >= self.width - 20:
            self.circle_vx *= -1
        
        if self.circle_y <= 20 or self.circle_y >= self.height - 20:
            self.circle_vy *= -1

    def continuous_capture(self):
        """Continuously capture images in a separate thread"""
        last_capture_time = time.time()
        
        while self.capture_running:
            current_time = time.time()
            
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                self.webcam_photo(self.circle_x, self.circle_y)
                last_capture_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent excessive CPU usage

    def webcam_photo(self, x, y):
        """Capture a photo from webcam and save with current circle position"""
        # Generate unique image filename with timestamp
        uid = f"{int(time.time() * 1000)}"  # Use milliseconds for more unique names
        img_filename = f"Dataset/{uid}.jpg"
        
        # Use the existing webcam connection
        ret, frame = self.cap.read()
        if ret:
            # Crop img to 1080x1080
            right_x = 1920 // 2 + 1080 // 2
            left_x = 1920 // 2 - 1080 // 2
            frame = frame[:, left_x:right_x, :]
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            # Save the image 
            cv2.imwrite(img_filename, frame)
            # Add row to dataframe
            norm_x = x / self.width
            norm_y = y / self.height
            new_row = pd.DataFrame([{"img_filename": img_filename, "x": norm_x, "y": norm_y}])
            self.temp_df = pd.concat([self.temp_df, new_row], ignore_index=True)
    
    def release_webcam(self):
        """Release the webcam resources"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def load_dataset_df(self):
        if os.path.exists("dataset.csv"):
            df = pd.read_csv("dataset.csv")
            new = False
        else:
            df = pd.DataFrame(columns=["img_filename", "x", "y"])
            new = True
        return df, new

    def save_dataset_df(self):
        # Concatenate the temp_df with the loaded df
        df, new = self.load_dataset_df()
        if new:
            df = self.temp_df
        else:
            df = pd.concat([df, self.temp_df], ignore_index=True)
        df.to_csv("dataset.csv", index=False)

if __name__ == "__main__":  
    dataset_generator = DatasetGenerator()
    try:
        dataset_generator.run_dataset_generator()
        dataset_generator.save_dataset_df()
    finally:
        dataset_generator.release_webcam()
        pygame.quit()