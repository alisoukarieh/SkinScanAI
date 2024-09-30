import pandas as pd
import shutil
import os

# Read the CSV data
df = pd.read_csv("./dataverse_files/HAM10000_metadata")

# Create directories for each unique dx value
dx_values = df['dx'].unique()
for dx in dx_values:
    os.makedirs(f"images/{dx}", exist_ok=True)

folder_paths = ["./dataverse_files/HAM10000_images_part_1", "./dataverse_files/HAM10000_images_part_2"]  # Add more paths as needed

# Iterate through the DataFrame and move images to corresponding directories
for index, row in df.iterrows():
    image_id = row['image_id']
    dx = row['dx']
    found = False
    for folder_path in folder_paths:
        source_path = f"./{folder_path}/{image_id}.jpg"  # Assuming image filenames have a .jpg extension
        if os.path.exists(source_path):
            target_path = f"images/{dx}/{image_id}.jpg"
            shutil.move(source_path, target_path)
            found = True
            break
    if not found:
        print(f"Image {image_id} not found in any specified folder.")

print("Images split successfully!")