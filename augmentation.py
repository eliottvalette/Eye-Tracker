"""
From images in dataset.csv, generate new images by applying random transformations.

dataset.csv, 
img_filename,x,y
Dataset/1751743551.jpg,0.0846666666666666,0.8821428571428571
Dataset/1751743552.jpg,0.4033333333333333,0.7976190476190477
Dataset/1751743553.jpg,0.4613333333333333,0.35

augmented_dataset.csv,
img_filename,x,y
Augmented_Dataset/1751743551_1.jpg,0.0846666666666666,0.8821428571428571
Augmented_Dataset/1751743551_2.jpg,0.0846666666666666,0.8821428571428571
...
Augmented_Dataset/1751743551_4.jpg,0.0846666666666666,0.8821428571428571
Augmented_Dataset/1751743552_1.jpg,0.4033333333333333,0.7976190476190477
Augmented_Dataset/1751743552_2.jpg,0.4033333333333333,0.7976190476190477
...
Augmented_Dataset/1751743552_4jpg,0.4033333333333333,0.7976190476190477
Augmented_Dataset/1751743553_1.jpg,0.4613333333333333,0.35
Augmented_Dataset/1751743553_2.jpg,0.4613333333333333,0.35
...
Augmented_Dataset/1751743553_4.jpg,0.4613333333333333,0.35
"""

import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import shutil

# Ensure TensorFlow is using the right backend
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_augmented_directory():
    """Create Augmented_Dataset directory if it doesn't exist"""
    if not os.path.exists("Augmented_Dataset"):
        os.makedirs("Augmented_Dataset")
    else:
        # Clean the directory
        shutil.rmtree("Augmented_Dataset")
        os.makedirs("Augmented_Dataset")

def clear_augmented_directory():
    """Delete all images in Augmented_Dataset directory"""
    if os.path.exists("Augmented_Dataset"):
        for filename in os.listdir("Augmented_Dataset"):
            file_path = os.path.join("Augmented_Dataset", filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Cleared all images from Augmented_Dataset directory")

def load_and_preprocess_image(image_path):
    """Load an image and convert it to a tensor"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    return img_tensor

def apply_augmentations(image_tensor):
    """Apply 8 different augmentations to an image"""
    augmented_images = []
    
    # 1. Original image (no augmentation)
    augmented_images.append(image_tensor)
    
    # Helper function for horizontal translation
    def horizontal_translation(img, pixels=20):
        # Create translation matrix
        translated = tf.roll(img, shift=pixels, axis=1)  # Shift horizontally
        return translated
    
    # 2. Saturation shift increase + horizontal shift right
    saturation_increase = tf.image.adjust_saturation(image_tensor, 1.5)
    saturation_increase_shifted = horizontal_translation(saturation_increase, 20)
    augmented_images.append(saturation_increase_shifted)

    # 3. Saturation shift decrease + horizontal shift left
    saturation_decrease = tf.image.adjust_saturation(image_tensor, 0.5)
    saturation_decrease_shifted = horizontal_translation(saturation_decrease, -20)
    augmented_images.append(saturation_decrease_shifted)    
    
    return augmented_images

def save_augmented_image(image_tensor, output_path):
    """Save a tensor image to disk"""
    # Convert tensor to numpy and ensure proper range
    image_np = image_tensor.numpy().astype(np.uint8)
    # Convert RGB to BGR for cv2
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)

def main():
    # Create or clean Augmented_Dataset directory
    create_augmented_directory()
    
    # Clear all existing images before adding new ones
    clear_augmented_directory()
    
    # Load original dataset
    original_dataset = pd.read_csv("dataset.csv")
    print(f"Loaded {len(original_dataset)} images from dataset.csv")
    
    # Create new dataframe for augmented dataset
    augmented_dataset = pd.DataFrame(columns=["img_filename", "x", "y"])
    
    # Process each image in the original dataset
    for idx, row in tqdm(original_dataset.iterrows(), total=len(original_dataset), desc="Augmenting images"):
        img_path = row['img_filename']
        x, y = row['x'], row['y']
        
        # Extract the base filename without the path and extension
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load the image
        try:
            image_tensor = load_and_preprocess_image(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        
        # Apply augmentations
        augmented_images = apply_augmentations(image_tensor)
        
        # Save each augmented image and add to the new dataframe
        for aug_idx, aug_image in enumerate(augmented_images, 1):
            # Create output path
            output_filename = f"Augmented_Dataset/{base_filename}_{aug_idx}.jpg"
            
            # Save the augmented image
            save_augmented_image(aug_image, output_filename)
            
            # Add to dataframe
            new_row = {"img_filename": output_filename, "x": x, "y": y}
            augmented_dataset = pd.concat([augmented_dataset, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the augmented dataset CSV
    augmented_dataset.to_csv("augmented_dataset.csv", index=False)
    print(f"Created {len(augmented_dataset)} augmented images")
    print(f"Augmented dataset saved to augmented_dataset.csv")

if __name__ == "__main__":
    main()
