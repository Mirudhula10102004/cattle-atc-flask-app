import os
import torch
import cv2
import matplotlib.pyplot as plt
from cattle_pose_dataset import CattlePoseDataset
from train_pose_model_resnet import ResNetPoseModel

def visualize_predictions(model_path, sample_image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_keypoints = 15
    model = ResNetPoseModel(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = CattlePoseDataset(root_dir='.', subset='val', augment=False, num_keypoints=num_keypoints)

    if sample_image not in dataset.image_files:
        print(f"'{sample_image}' not found in validation dataset. Using first image instead.")
        sample_image = dataset.image_files[0]

    idx = dataset.image_files.index(sample_image)
    image, true_heatmaps = dataset[idx]
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmaps = model(input_image).cpu().squeeze(0)

    pred_points = []
    for i in range(num_keypoints):
        heatmap = pred_heatmaps[i]
        max_val = heatmap.max()
        if max_val == 0:
            x = y = torch.tensor([0])
        else:
            max_pos = (heatmap == max_val).nonzero()
            y, x = max_pos[0][0], max_pos[0][1]
        x_coord = x.item() * (dataset.image_size / dataset.heatmap_size)
        y_coord = y.item() * (dataset.image_size / dataset.heatmap_size)
        pred_points.append((x_coord, y_coord))

    gt_points = []
    for i in range(num_keypoints):
        heatmap = true_heatmaps[i]
        max_val = heatmap.max()
        if max_val == 0:
            x = y = torch.tensor([0])
        else:
            max_pos = (heatmap == max_val).nonzero()
            y, x = max_pos[0][0], max_pos[0][1]
        x_coord = x.item() * (dataset.image_size / dataset.heatmap_size)
        y_coord = y.item() * (dataset.image_size / dataset.heatmap_size)
        gt_points.append((x_coord, y_coord))

    image_np = image.permute(1, 2, 0).numpy()

    plt.imshow(image_np)
    for (x, y) in gt_points:
        if x is not None and y is not None:
            plt.scatter(x, y, c='red', s=50, marker='o', label='Ground Truth')
    for (x, y) in pred_points:
        if x is not None and y is not None:
            plt.scatter(x, y, c='lime', s=40, marker='x', label='Prediction')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"Ground Truth (red) vs Prediction (green): {sample_image}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    model_checkpoint_path = 'model_checkpoint_resnet_epoch20.pth'
    sample_img = ''  # Leave blank to auto-select first val image
    visualize_predictions(model_checkpoint_path, sample_img)
