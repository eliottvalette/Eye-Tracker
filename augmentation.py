import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
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

    rd_float = rd.random()
    
    if rd_float < 0.2:
        aug_img = img.copy()
    elif rd_float < 0.75:
        # Augmentation de luminosité avec OpenCV
        beta = rd.uniform(-50, 50)
        aug_img = cv2.convertScaleAbs(img, beta=beta)
    else:
        # shift lateral avec bordure miroir
        shift = rd.uniform(-10, 10)
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        aug_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), 
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
