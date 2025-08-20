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
import numpy as np
import random as rd
import os
import pandas as pd
import cv2
import time
import threading
import math

TIME = 120
CAPTURE_INTERVAL = 0.1  # Capture every 100ms

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
        self.circle_vx = rd.uniform(100, 200) * rd.choice([-1, 1])  # Random velocity
        self.circle_vy = rd.uniform(100, 200) * rd.choice([-1, 1])
        
        # Advanced movement properties
        self.time_start = time.time()
        self.base_speed = rd.uniform(80, 150)
        self.curve_amplitude = rd.uniform(50, 100)
        self.velocity_change_interval = rd.uniform(2.0, 4.0)  # Change velocity every 2-4 seconds
        self.last_velocity_change = time.time()
        self.movement_pattern = rd.choice(['bounce', 'curve', 'spiral', 'random_walk'])
        
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
        current_time = time.time()
        elapsed_time = current_time - self.time_start
        
        # Change velocity periodically
        if current_time - self.last_velocity_change > self.velocity_change_interval:
            self.change_velocity()
            self.last_velocity_change = current_time
        
        # Apply different movement patterns
        if self.movement_pattern == 'bounce':
            self.update_circle_position_bounce()
        elif self.movement_pattern == 'curve':
            self.update_circle_position_curve(elapsed_time)
        elif self.movement_pattern == 'spiral':
            self.update_circle_position_spiral(elapsed_time)
        elif self.movement_pattern == 'random_walk':
            self.update_circle_position_random_walk(elapsed_time)
        
        # Keep circle within bounds
        self.circle_x = max(20, min(self.width - 20, self.circle_x))
        self.circle_y = max(20, min(self.height - 20, self.circle_y))

    def update_circle_position_bounce(self):
        """Update circle position with bouncing off walls"""
        dt = 1/60  # Assuming 60 FPS
        self.circle_x += self.circle_vx * dt
        self.circle_y += self.circle_vy * dt
        
        # Bounce off walls
        if self.circle_x <= 20 or self.circle_x >= self.width - 20:
            self.circle_vx = -self.circle_vx
            self.circle_x = max(20, min(self.width - 20, self.circle_x))
            
        if self.circle_y <= 20 or self.circle_y >= self.height - 20:
            self.circle_vy = -self.circle_vy
            self.circle_y = max(20, min(self.height - 20, self.circle_y))

    def update_circle_position_curve(self, elapsed_time):
        """Update circle position with curved movement using sin and cos"""
        dt = 1/60
        
        # Base movement with stronger velocity
        self.circle_x += self.circle_vx * dt
        self.circle_y += self.circle_vy * dt
        
        # Add curved motion with reduced amplitude to avoid convergence
        curve_x = rd.uniform(-self.curve_amplitude, self.curve_amplitude)
        curve_y = rd.uniform(-self.curve_amplitude, self.curve_amplitude)
        
        self.circle_x += curve_x * dt
        self.circle_y += curve_y * dt
        
        # Bounce off walls
        if self.circle_x <= 20 or self.circle_x >= self.width - 20:
            self.circle_vx = -self.circle_vx
            self.circle_x = max(20, min(self.width - 20, self.circle_x))
            
        if self.circle_y <= 20 or self.circle_y >= self.height - 20:
            self.circle_vy = -self.circle_vy
            self.circle_y = max(20, min(self.height - 20, self.circle_y))

    def update_circle_position_spiral(self, elapsed_time):
        """Update circle position with spiral movement"""
        dt = 1/60
        
        # Spiral center - use current position as base to avoid convergence
        base_x = self.circle_x
        base_y = self.circle_y
        
        # Spiral parameters with larger radius and faster movement
        spiral_radius = 100 + 80 * math.sin(elapsed_time * 0.3)
        spiral_angle = elapsed_time * 3
        
        # Calculate spiral offset from current position
        spiral_offset_x = spiral_radius * math.cos(spiral_angle)
        spiral_offset_y = spiral_radius * math.sin(spiral_angle)
        
        # Add spiral motion to current position
        self.circle_x += spiral_offset_x * dt * 0.5
        self.circle_y += spiral_offset_y * dt * 0.5
        
        # Add some random drift to prevent staying in one area
        drift_x = 20 * math.sin(elapsed_time * 1.7) * math.cos(elapsed_time * 0.9)
        drift_y = 20 * math.cos(elapsed_time * 1.3) * math.sin(elapsed_time * 1.1)
        
        self.circle_x += drift_x * dt
        self.circle_y += drift_y * dt

    def update_circle_position_random_walk(self, elapsed_time):
        """Update circle position with random walk movement"""
        dt = 1/60
        
        # Random walk step size
        step_size = 3.0
        
        # Add random movement in both directions
        random_x = rd.uniform(-step_size, step_size)
        random_y = rd.uniform(-step_size, step_size)
        
        self.circle_x += random_x
        self.circle_y += random_y
        
        # Add some momentum to make movement more natural
        momentum_x = rd.uniform(-1.0, 1.0) * step_size * 0.5
        momentum_y = rd.uniform(-1.0, 1.0) * step_size * 0.5
        
        self.circle_x += momentum_x
        self.circle_y += momentum_y
        
        # Bounce off walls
        if self.circle_x <= 20 or self.circle_x >= self.width - 20:
            self.circle_x = max(20, min(self.width - 20, self.circle_x))
            
        if self.circle_y <= 20 or self.circle_y >= self.height - 20:
            self.circle_y = max(20, min(self.height - 20, self.circle_y))

    def change_velocity(self):
        """Change velocity randomly"""
        # Random velocity change
        self.circle_vx = rd.uniform(80, 200) * rd.choice([-1, 1])
        self.circle_vy = rd.uniform(80, 200) * rd.choice([-1, 1])
        
        # Randomly change movement pattern
        if rd.random() < 0.3:  # 30% chance to change pattern
            self.movement_pattern = rd.choice(['bounce', 'curve', 'spiral', 'random_walk'])
            self.curve_amplitude = rd.uniform(50, 100)
        
        # Update velocity change interval
        self.velocity_change_interval = rd.uniform(2.0, 4.0)

    def update_circle_position(self):
        """Legacy method - kept for compatibility"""
        self.update_circle_position_bounce()

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