"""
Rank All Predictions

This script ranks all predictions by the distance to the true coordinates.
It then saves the ranked predictions to a CSV file.
"""
import torch
import numpy as np
from torchvision import transforms
from model import Model
from PIL import Image
import os
import pandas as pd
import glob
from tqdm import tqdm

class Ranker:
    def __init__(self, model_path="best_model.pth"):
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        self.model.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform for RGB input (consistent with training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def rank_predictions(self, dataset_path, model_path="best_model.pth"):
        """
        Analyze and visualize activation maps for random samples from the dataset
        """
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model()
        model.load_model(model_path)
        model.to(device)
        model.eval()
        
        # Define transform for RGB input
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get list of image files
        image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
        
        if not image_files:
            print(f"No images found in {dataset_path}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Load augmented dataset to get true coordinates
        dataset_df = pd.read_csv("dataset.csv")
        
        # Initialize list to store predictions
        predictions_list = []
        
        for i, img_path in tqdm(enumerate(image_files), desc="Processing images", unit="img", total=len(image_files)):
            # Load image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            
            # Transform image
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                predictions = model(img_tensor)
            
            pred_x = predictions[0, 0].item()
            pred_y = predictions[0, 1].item()
            
            # Get true coordinates from dataset
            img_filename = os.path.basename(img_path)
            matching_rows = dataset_df[dataset_df['img_filename'].str.contains(img_filename)]
            if len(matching_rows) > 0:
                true_x = matching_rows.iloc[0]['x']
                true_y = matching_rows.iloc[0]['y']
            else:
                true_x, true_y = 0.5, 0.5  # fallback
            
            # Calculate distance to true coordinates
            distance = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            
            # Add to list of predictions
            predictions_list.append({
                'img_path': img_path,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'true_x': true_x,
                'true_y': true_y,
                'distance': distance
            })

        # Sort predictions by distance
        predictions_list.sort(key=lambda x: x['distance'], reverse=True)
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame(predictions_list)
        predictions_df.to_csv('predictions.csv', index=False)
        
        print(f"Processed {len(predictions_list)} images")
        print(f"Saved predictions to predictions.csv")
        
        # Clean up
        model.cpu()
        return predictions_df

if __name__ == "__main__":
    ranker = Ranker()
    predictions_df = ranker.rank_predictions("Dataset")
    print(predictions_df)