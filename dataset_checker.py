import pandas as pd
import os

# verify that each image list in the df exist in the Dataset folder
print("Checking original dataset...")
raw_df = pd.read_csv("dataset.csv")
exist_count = 0
not_exist_count = 0

for index, row in raw_df.iterrows():
    if not os.path.exists(row["img_filename"]):
        print(f"Image {row['img_filename']} does not exist")
        not_exist_count += 1
    else:
        exist_count += 1

print(f"Exist count: {exist_count}")
print(f"Not exist count: {not_exist_count}")

print("Checking augmented dataset...")
aug_df = pd.read_csv("augmented_dataset.csv")
exist_count_augmented = 0
not_exist_count_augmented = 0

for index, row in aug_df.iterrows():
    if not os.path.exists(row["img_filename"]):
        print(f"Image {row['img_filename']} does not exist")
        not_exist_count_augmented += 1
    else:
        exist_count_augmented += 1

print(f"Exist count augmented: {exist_count_augmented}")
print(f"Not exist count augmented: {not_exist_count_augmented}")



print("\n---- Same but Reverse ----\n")

os.makedirs("not_listed", exist_ok=True)
dataset_raw_images = os.listdir('Dataset')
dataset_raw_images.sort()
images_listed_in_csv = raw_df['img_filename'].tolist()
not_listed_counter = 0
listed_counter = 0
not_listed_images = []

for raw_img_filename in dataset_raw_images :
    if not f"Dataset/{raw_img_filename}" in images_listed_in_csv :
        print(f"Image {raw_img_filename} is not listed in the dataset")
        not_listed_images.append(raw_img_filename)
        not_listed_counter += 1
    else :
        listed_counter +=1

print(f"Listed count: {listed_counter}")
print(f"Not Listed count  : {not_listed_counter}")

# Delete unlisted images
if not_listed_images:
    print(f"\nFound {len(not_listed_images)} unlisted images to delete.")
    response = input("Do you want to delete these unlisted images? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        deleted_count = 0
        for img_filename in not_listed_images:
            img_path = os.path.join('Dataset', img_filename)
            try:
                os.remove(img_path)
                print(f"Deleted: {img_filename}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {img_filename}: {e}")
        
        print(f"\nSuccessfully deleted {deleted_count} out of {len(not_listed_images)} unlisted images.")
    else:
        print("Deletion cancelled.")
else:
    print("No unlisted images found to delete.")