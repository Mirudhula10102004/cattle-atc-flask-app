import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class CattlePoseDataset(Dataset):
    def __init__(self, root_dir, subset='train', image_size=256, heatmap_size=64, sigma=2, num_keypoints=15, augment=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.num_keypoints = num_keypoints
        self.augment = augment

        self.image_dir = os.path.join(root_dir, 'images', subset)
        self.label_dir = os.path.join(root_dir, 'labels', subset)

        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.image_files.sort()

        # Define augmentation transforms
        if self.augment:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Apply transforms (augmentations)
        image = self.transform(image)

        # Create heatmaps
        heatmaps = self.create_heatmaps(label_path)

        if heatmaps.shape[0] != self.num_keypoints:
            print(f"Warning: Number of heatmaps {heatmaps.shape[0]} != expected {self.num_keypoints} for {img_name}")

        return image, heatmaps

    def create_heatmaps(self, label_path):
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_size, self.heatmap_size), dtype=np.float32)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            kp_index = int(parts[0])
            if kp_index >= self.num_keypoints:
                continue
            x_center = float(parts[1])
            y_center = float(parts[2])
            x = x_center * self.heatmap_size
            y = y_center * self.heatmap_size
            heatmaps[kp_index] = self.put_gaussian_heatmap(heatmaps[kp_index], x, y, self.sigma)

        return torch.from_numpy(heatmaps)

    @staticmethod
    def put_gaussian_heatmap(heatmap, center_x, center_y, sigma):
        tmp_size = sigma * 3
        ul = [int(center_x - tmp_size), int(center_y - tmp_size)]
        br = [int(center_x + tmp_size), int(center_y + tmp_size)]
        size = heatmap.shape[0]
        if ul[0] >= size or ul[1] >= size or br[0] < 0 or br[1] < 0:
            return heatmap
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        xx = x - center_x
        yy = y - center_y
        gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], size) - ul[1]
        img_x = max(0, ul[0]), min(br[0], size)
        img_y = max(0, ul[1]), min(br[1], size)
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            gaussian[g_y[0]:g_y[1], g_x[0]:g_x[1]],
        )
        return heatmap
