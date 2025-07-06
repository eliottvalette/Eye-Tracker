import pandas as pd
import os

# verify that each image list in the df exist in the Dataset folder
print("Checking original dataset...")
df = pd.read_csv("dataset.csv")
exist_count = 0
not_exist_count = 0

for index, row in df.iterrows():
    if not os.path.exists(row["img_filename"]):
        print(f"Image {row['img_filename']} does not exist")
        not_exist_count += 1
    else:
        exist_count += 1

print(f"Exist count: {exist_count}")
print(f"Not exist count: {not_exist_count}")

print("Checking augmented dataset...")
df = pd.read_csv("augmented_dataset.csv")
exist_count_augmented = 0
not_exist_count_augmented = 0

for index, row in df.iterrows():
    if not os.path.exists(row["img_filename"]):
        print(f"Image {row['img_filename']} does not exist")
        not_exist_count_augmented += 1
    else:
        exist_count_augmented += 1

print(f"Exist count augmented: {exist_count_augmented}")
print(f"Not exist count augmented: {not_exist_count_augmented}")