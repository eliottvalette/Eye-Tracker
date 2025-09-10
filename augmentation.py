import os
import pandas as pd
import cv2
import numpy as np
import random as rd

# Créer le dossier pour les images augmentées
os.makedirs("Augmented_Dataset", exist_ok=True)

# Vider le dossier Augmented_Dataset
for file in os.listdir("Augmented_Dataset"):
    os.remove(os.path.join("Augmented_Dataset", file))

# Charger le dataset original
df = pd.read_csv("dataset.csv")
print(f"Dataset chargé: {len(df)} images")

# Liste pour stocker les nouvelles données
new_data = []

# Traiter chaque image
for idx, row in df.iterrows():
    img_path = row['img_filename']
    x, y = row['x'], row['y']
    
    # Nom du fichier sans le dossier
    filename = os.path.basename(img_path)
    img_full_path = img_path
    
    # Charger l'image
    img = cv2.imread(img_full_path)

    # All augmentations have the same probability to happen
    aug_type = rd.randint(0, 6)
    if aug_type == 0:
        # No augmentation
        aug_img = img.copy()
    elif aug_type == 1:
        # Brightness/contrast
        alpha = rd.uniform(0.3, 2.5)  # Contrast
        beta = rd.uniform(-30, 30)    # Brightness
        aug_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    elif aug_type == 2:
        # Horizontal shift
        shift = rd.uniform(-30, 30)
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        aug_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), 
                                borderMode=cv2.BORDER_REFLECT_101)
    elif aug_type == 3:
        # Vertical shift
        shift = rd.uniform(-20, 20)
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        aug_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), 
                                borderMode=cv2.BORDER_REFLECT_101)
    elif aug_type == 4:
        # Blur
        aug_img = cv2.GaussianBlur(img, (3, 3), 0)
    elif aug_type == 5:
        # Color channel manipulation
        b, g, r = cv2.split(img)
        channel_mult = rd.uniform(0.5, 1.8)
        channel_idx = rd.randint(0, 2)
        channels = [b, g, r]
        channels[channel_idx] = cv2.multiply(channels[channel_idx], channel_mult)
        aug_img = cv2.merge(channels)
    elif aug_type == 6:
        # Perspective transformation
        h, w = img.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = rd.uniform(-0.1, 0.1)
        pts2 = np.float32([[offset*w, offset*h], 
                          [w-offset*w, offset*h], 
                          [offset*w, h-offset*h], 
                          [w-offset*w, h-offset*h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        aug_img = cv2.warpPerspective(img, M, (w, h), 
                                    borderMode=cv2.BORDER_REFLECT_101)
        
    # Créer le nom du fichier
    aug_save_path = f"Augmented_Dataset/{filename}"
    
    # Sauvegarder l'image augmentée
    cv2.imwrite(aug_save_path, aug_img)
    
    # Ajouter aux données
    new_data.append({
        'img_filename': aug_save_path,
        'x': x,
        'y': y
    })

# Créer le nouveau CSV
new_df = pd.DataFrame(new_data)
new_df.to_csv("augmented_dataset.csv", index=False)

print(f"Terminé! {len(new_df)} images créées")
print(f"Images sauvegardées dans: Augmented_Dataset/")
print(f"CSV sauvegardé: augmented_dataset.csv")
