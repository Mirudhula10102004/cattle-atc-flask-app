import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cattle_pose_dataset import CattlePoseDataset


class SimplePoseModel(nn.Module):
    def __init__(self, num_keypoints):
        super(SimplePoseModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, num_keypoints, 1)  # Output keypoint heatmaps
        )
    def forward(self, x):
        return self.features(x)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set num_keypoints to match your dataset
    num_keypoints = 15 

    train_set = CattlePoseDataset(root_dir='.', subset='train', augment=True, num_keypoints=num_keypoints)
    val_set = CattlePoseDataset(root_dir='.', subset='val', augment=False, num_keypoints=num_keypoints)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = SimplePoseModel(num_keypoints=num_keypoints).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, heatmaps in train_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.6f} Val Loss: {val_loss/len(val_loader):.6f}")

        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), f'model_checkpoint_epoch{epoch+1}.pth')

if __name__ == '__main__':
    train()
