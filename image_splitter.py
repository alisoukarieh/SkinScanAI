import splitfolders
import os

source_dir = "images"
output_dir = os.path.join(source_dir, "../Dataset")

# Split the images in each subdirectory into 80% training and 20% validation
splitfolders.ratio(source_dir, output=output_dir, ratio=(0.8, 0.2), seed=42)
