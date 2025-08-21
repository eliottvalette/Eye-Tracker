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
import time
import math

TIME = 120

class DatasetGenerator:
    def __init__(self, width=1500, height=840):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dataset Generator")
        self.clock = pygame.time.Clock()
        self.temp_df = pd.DataFrame(columns=["img_filename", "x", "y"])
        
        # Circle movement properties
        self.circle_x = width // 2
        self.circle_y = height // 2
        self.circle_vx = rd.choice([-1, 1]) * rd.uniform(6, 8)
        self.circle_vy = rd.choice([-1, 1]) * rd.uniform(6, 8)
        
    def run_dataset_generator(self):
        start_time = time.time()
        
        while time.time() - start_time < TIME:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
        
    def update_circle_position_advanced(self):
        """Update circle position with advanced movement patterns"""
        self.circle_x += self.circle_vx
        self.circle_y += self.circle_vy
        
        # Check if circle is hitting the edges
        if self.circle_x <= 20 or self.circle_x >= self.width - 20:
            self.circle_vx *= -1
        
        if self.circle_y <= 20 or self.circle_y >= self.height - 20:
            self.circle_vy *= -1

if __name__ == "__main__":  
    dataset_generator = DatasetGenerator()
    try:
        dataset_generator.run_dataset_generator()
    finally:
        pygame.quit()