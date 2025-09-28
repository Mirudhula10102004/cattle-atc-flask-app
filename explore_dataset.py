print("Running script:", __file__)

import os

root_dir = '.'  # current folder

# Subfolders for images and labels inside the dataset
image_subfolders = ['images/train', 'images/val']
label_subfolders = ['labels/train', 'labels/val']

def count_files(folders, extensions):
    total_files = 0
    samples = []
    for folder in folders:
        path = os.path.join(root_dir, folder)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(extensions)]
            total_files += len(files)
            samples.extend(files[:3])  # Collect first 3 files for sample
        else:
            print(f"Warning: Folder not found {path}")
    return total_files, samples

image_count, image_samples = count_files(image_subfolders, ('.jpg', '.png'))
label_count, label_samples = count_files(label_subfolders, ('.txt',))  # Assuming labels are .txt files

print(f"Total images (train + val): {image_count}")
print(f"Sample image files: {image_samples}")
print(f"Total annotation files (train + val): {label_count}")
print(f"Sample annotation files: {label_samples}")
