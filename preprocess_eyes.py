import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def get_eye_crop(image, keypoint, width=41, height=31):
    """Crop eye given center keypoint. Fallback to black image if out of bounds/invalid."""
    fallback = np.zeros((height, width, 3), dtype=np.uint8)
    if keypoint[0] < 0 or keypoint[1] < 0:
        return fallback, -1.0, -1.0
    
    x, y = int(keypoint[0]), int(keypoint[1])
    half_w, half_h = width // 2, height // 2
    
    x1, y1 = x - half_w, y - half_h
    x2, y2 = x + half_w + (width % 2), y + half_h + (height % 2)
    
    h_img, w_img = image.shape[:2]
    
    if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
        return fallback, -1.0, -1.0
        
    crop = image[y1:y2, x1:x2]
    
    # Normalize coords for the model
    norm_x = x / w_img
    norm_y = y / h_img
    
    return crop, norm_x, norm_y

def main():
    print("Loading YOLOv8-pose...")
    model = YOLO("yolov8n-pose.pt")
    
    # Priority: augmented dataset, then base dataset
    base_csv = "augmented_dataset.csv" if os.path.exists("augmented_dataset.csv") else "dataset.csv"
    if not os.path.exists(base_csv):
        raise FileNotFoundError(f"No {base_csv} found.")
        
    df = pd.read_csv(base_csv)
    os.makedirs("Dataset_Eyes", exist_ok=True)
    
    new_rows = []
    print(f"Preprocessing {len(df)} images offline (YOLO Fallback on missing eyes)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['img_filename']
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        results = model(image, verbose=False)
        
        lx, ly, rx, ry = -1.0, -1.0, -1.0, -1.0
        left_eye_crop = np.zeros((31, 41, 3), dtype=np.uint8)
        right_eye_crop = np.zeros((31, 41, 3), dtype=np.uint8)
        
        if len(results) > 0 and len(results[0].keypoints) > 0:
            # Check if keypoints are present
            if hasattr(results[0].keypoints, 'xy') and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                if len(keypoints) >= 3:
                    # keypoint 1: Left Eye, 2: Right Eye
                    l_kp = keypoints[1]
                    r_kp = keypoints[2]
                    
                    left_eye_crop, lx, ly = get_eye_crop(image, l_kp)
                    right_eye_crop, rx, ry = get_eye_crop(image, r_kp)
                
        # Save crops
        uid = os.path.splitext(os.path.basename(img_path))[0]
        left_path = f"Dataset_Eyes/{uid}_left.jpg"
        right_path = f"Dataset_Eyes/{uid}_right.jpg"
        
        cv2.imwrite(left_path, left_eye_crop)
        cv2.imwrite(right_path, right_eye_crop)
        
        new_row = row.copy()
        new_row['left_eye_img'] = left_path
        new_row['right_eye_img'] = right_path
        new_row['lx'] = lx
        new_row['ly'] = ly
        new_row['rx'] = rx
        new_row['ry'] = ry
        new_rows.append(new_row)
        
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv("dataset_eyes.csv", index=False)
    print("Preprocessed dataset saved to dataset_eyes.csv")

if __name__ == "__main__":
    main()
