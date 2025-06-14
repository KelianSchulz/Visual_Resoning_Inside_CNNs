import os
import csv

root_dir = "heatmap_concepts"
output_csv = "heatmap_concepts_dataset.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "concept_label"])

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".png"):
                    image_path = os.path.join(folder_path, filename)
                    writer.writerow([image_path, folder])
