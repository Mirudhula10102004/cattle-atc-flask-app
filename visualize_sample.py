import os
import cv2
import matplotlib.pyplot as plt

# Paths
root_dir = '.'
image_folder = os.path.join(root_dir, 'images/train')
label_folder = os.path.join(root_dir, 'labels/train')

# Select a sample image file from previous output or directory
sample_image = '11_b4-1_r_85_F.jpg'  # replace with your image filename if you want

# Derive corresponding label filename
label_file = sample_image.replace('.jpg', '.txt')

# Load image
image_path = os.path.join(image_folder, sample_image)
image = cv2.imread(image_path)

if image is None:
    print("Image not found:", image_path)
    exit()

# Convert color to RGB for matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load annotation points from label file
label_path = os.path.join(label_folder, label_file)
if not os.path.exists(label_path):
    print("Label file not found:", label_path)
    exit()

with open(label_path, 'r') as f:
    lines = f.readlines()

# Each line expected: <class> <x_center> <y_center> <width> <height>
# We'll extract points assuming YOLO format normalized coords

img_h, img_w, _ = image.shape
points = []

for line in lines:
    parts = line.strip().split()
    if len(parts) >= 5:
        # Extract normalized center x and y
        x_center = float(parts[1]) * img_w
        y_center = float(parts[2]) * img_h
        points.append((x_center, y_center))

# Plot image and annotations
plt.imshow(image)
plt.scatter([p[0] for p in points], [p[1] for p in points], s=50, c='r', marker='o')
plt.title(sample_image)
plt.axis('off')
plt.show()
