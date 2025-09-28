import torch
import numpy as np
import csv
from cattle_pose_dataset import CattlePoseDataset
from train_pose_model_resnet import ResNetPoseModel

def euclidean_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_body_length(keypoints):
    return euclidean_distance(keypoints[0], keypoints[1])

def calculate_height_at_withers(keypoints, image_height=224):
    if keypoints[2] is None:
        return None
    return image_height - keypoints[2][1]

def calculate_chest_width(keypoints):
    return euclidean_distance(keypoints[3], keypoints[4])

def calculate_rump_angle(keypoints):
    p4, p5, p6 = keypoints[4], keypoints[5], keypoints[6]
    if None in (p4, p5, p6):
        return None
    v1 = np.array(p4) - np.array(p5)
    v2 = np.array(p6) - np.array(p5)
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)

def extract_measurements_from_keypoints(keypoints):
    return {
        "body_length": calculate_body_length(keypoints),
        "height_at_withers": calculate_height_at_withers(keypoints),
        "chest_width": calculate_chest_width(keypoints),
        "rump_angle": calculate_rump_angle(keypoints),
    }

def run_and_save_results(model_path, save_csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_keypoints = 15
    dataset = CattlePoseDataset(root_dir='.', subset='val', augment=False, num_keypoints=num_keypoints)
    model = ResNetPoseModel(num_keypoints=num_keypoints).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(save_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['image_name', 'body_length', 'height_at_withers', 'chest_width', 'rump_angle']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, image_name in enumerate(dataset.image_files):
            image, _ = dataset[idx]
            input_image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_heatmaps = model(input_image).cpu().squeeze(0)

            pred_keypoints = []
            for i in range(num_keypoints):
                heatmap = pred_heatmaps[i]
                max_val = heatmap.max()
                if max_val == 0:
                    x = y = None
                else:
                    max_pos = (heatmap == max_val).nonzero()
                    y, x = max_pos[0][0].item(), max_pos[0][1].item()
                if x is not None and y is not None:
                    x_coord = x * (dataset.image_size / dataset.heatmap_size)
                    y_coord = y * (dataset.image_size / dataset.heatmap_size)
                    pred_keypoints.append((x_coord, y_coord))
                else:
                    pred_keypoints.append((None, None))

            # Debug print for keypoints
            print(f"Image: {image_name} Predicted keypoints:")
            for i, kp in enumerate(pred_keypoints):
                print(f"  Keypoint {i}: {kp}")

            measurements = extract_measurements_from_keypoints(pred_keypoints)
            row = {'image_name': image_name}
            row.update(measurements)
            writer.writerow(row)
            print(f"Processed {image_name} with measurements {measurements}")

if __name__ == '__main__':
    run_and_save_results('model_checkpoint_resnet_epoch20.pth', 'animal_measurements_debug.csv')
