import pygame
import numpy as np
import random as rd
import os
import pandas as pd
import cv2
import time

class DatasetGenerator:
    def __init__(self, width=1500, height=840):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dataset Generator")
        self.clock = pygame.time.Clock()
        self.photo_taken = False
        self.temp_df = pd.DataFrame(columns=["img_filename", "x", "y"])
    
    def run_dataset_generator(self):
        start_time = time.time()
        old_sample_idx = -1
        # 10 seconds
        while time.time() - start_time < 6:
            sample_idx = (time.time() - start_time) // 2
            time_remaining_before_photo = (time.time() - start_time) % 2

            if sample_idx != old_sample_idx:
                self.photo_taken = False
                x, y = rd.randint(20, self.width - 20), rd.randint(20, self.height - 20)
            
            # Fill the screen with black
            self.screen.fill((0, 0, 0))
            
            # Draw text on the screen
            font = pygame.font.Font(None, 96)
            text = font.render(f"{time_remaining_before_photo:.1f}s", True, (255, 255, 255))
            self.screen.blit(text, (self.width / 2, self.height / 2))

            # Draw a circle on the screen
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 20)
            
            pygame.display.flip()

            # If time_remaining_before_photo is 0, take a photo
            if time_remaining_before_photo <= 0.1 and not self.photo_taken:
                self.webcam_photo(x, y)
                self.photo_taken = True

            old_sample_idx = sample_idx
        

    def webcam_photo(self, x, y):
        # Generate unique image filename with timestamp and sample index
        uid = f"{int(time.time())}"
        img_filename = f"Dataset/{uid}.jpg"
        
        # Capture image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            # Save the image 
            cv2.imwrite(img_filename, frame)
            # Add row to dataframe
            norm_x = x / self.width
            norm_y = y / self.height
            new_row = pd.DataFrame([{"img_filename": img_filename, "x": norm_x, "y": norm_y}])
            self.temp_df = pd.concat([self.temp_df, new_row], ignore_index=True)
            
        cap.release()

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
    dataset_generator.run_dataset_generator()
    dataset_generator.save_dataset_df()
    pygame.quit()