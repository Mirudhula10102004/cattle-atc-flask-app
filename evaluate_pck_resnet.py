import torch
from cattle_pose_dataset import CattlePoseDataset
from train_pose_model_resnet import ResNetPoseModel
import numpy as np

def compute_pck(pred_points, gt_points, threshold):
    correct = 0
    total = 0
    for p, g in zip(pred_points, gt_points):
        if p[0] is None or g[0] is None:
            continue
        dist = np.linalg.norm(np.array(p) - np.array(g))
        if dist <= threshold:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def evaluate_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_keypoints = 15
    dataset = CattlePoseDataset(root_dir='.', subset='val', augment=False, num_keypoints=num_keypoints)
    model = ResNetPoseModel(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    threshold = 10  # Pixel threshold for correctness

    total_pck = 0
    count = 0

    for i in range(len(dataset)):
        image, true_heatmaps = dataset[i]
        input_image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_heatmaps = model(input_image).cpu().squeeze(0)

        pred_points = []
        gt_points = []

        for j in range(num_keypoints):
            heatmap = pred_heatmaps[j]
            max_val = heatmap.max()
            if max_val == 0:
                x = y = torch.tensor([0])
            else:
                max_pos = (heatmap == max_val).nonzero()
                y, x = max_pos[0][0], max_pos[0][1]
            x_coord = x.item() * (dataset.image_size / dataset.heatmap_size)
            y_coord = y.item() * (dataset.image_size / dataset.heatmap_size)
            pred_points.append((x_coord, y_coord))

            heatmap = true_heatmaps[j]
            max_val = heatmap.max()
            if max_val == 0:
                x = y = torch.tensor([0])
            else:
                max_pos = (heatmap == max_val).nonzero()
                y, x = max_pos[0][0], max_pos[0][1]
            x_coord = x.item() * (dataset.image_size / dataset.heatmap_size)
            y_coord = y.item() * (dataset.image_size / dataset.heatmap_size)
            gt_points.append((x_coord, y_coord))

        pck = compute_pck(pred_points, gt_points, threshold)
        total_pck += pck
        count += 1

    avg_pck = total_pck / count if count > 0 else 0
    print(f"Average PCK@{threshold}px: {avg_pck:.4f}")

if __name__ == '__main__':
    evaluate_model('model_checkpoint_resnet_epoch20.pth')
