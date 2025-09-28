import torch
import matplotlib.pyplot as plt
import cv2
from cattle_pose_dataset import CattlePoseDataset
from train_pose_model_resnet import ResNetPoseModel

def visualize_heatmaps_and_keypoints(model_path, num_display=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_keypoints = 15

    dataset = CattlePoseDataset(root_dir='.', subset='val', augment=False, num_keypoints=num_keypoints)
    model = ResNetPoseModel(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for i in range(num_display):
        image, true_heatmaps = dataset[i]
        input_image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_heatmaps = model(input_image).cpu().squeeze(0)

        image_np = image.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(2, num_keypoints // 2 + 1, figsize=(15, 6))
        axes = axes.flatten()

        for j in range(num_keypoints):
            heatmap = pred_heatmaps[j].numpy()
            axes[j].imshow(heatmap, cmap='jet')
            axes[j].axis('off')
            axes[j].set_title(f'Heatmap {j}')

        plt.suptitle(f"Predicted Heatmaps - Image {dataset.image_files[i]}")
        plt.show()

        # Overlay keypoints on image
        keypoints = []
        for j in range(num_keypoints):
            heatmap = pred_heatmaps[j]
            max_val = heatmap.max()
            if max_val == 0:
                keypoints.append(None)
                continue
            max_pos = (heatmap == max_val).nonzero()
            y, x = max_pos[0][0].item(), max_pos[0][1].item()
            x_coord = x * (dataset.image_size / dataset.heatmap_size)
            y_coord = y * (dataset.image_size / dataset.heatmap_size)
            keypoints.append((int(x_coord), int(y_coord)))

        image_disp = (image_np * 255).astype('uint8').copy()

        for point in keypoints:
            if point is not None:
                cv2.circle(image_disp, point, radius=5, color=(0, 255, 0), thickness=-1)

        plt.imshow(image_disp)
        plt.title(f"Predicted Keypoints Overlay - Image {dataset.image_files[i]}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    visualize_heatmaps_and_keypoints('model_checkpoint_resnet_epoch20.pth')
