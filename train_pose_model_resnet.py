import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from cattle_pose_dataset import CattlePoseDataset

class ResNetPoseModel(nn.Module):
    def __init__(self, num_keypoints):
        super(ResNetPoseModel, self).__init__()
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool & fc
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # Upsample to 64x64
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_keypoints = 15

    train_set = CattlePoseDataset(root_dir='.', subset='train', augment=True, num_keypoints=num_keypoints)
    val_set = CattlePoseDataset(root_dir='.', subset='val', augment=False, num_keypoints=num_keypoints)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = ResNetPoseModel(num_keypoints=num_keypoints).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 20
    for epoch in range(epochs):
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
        
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, heatmaps in val_loader:
                images, heatmaps = images.to(device), heatmaps.to(device)
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.6f} Val Loss: {val_loss/len(val_loader):.6f}")
        torch.save(model.state_dict(), f'model_checkpoint_resnet_epoch{epoch+1}.pth')

if __name__ == '__main__':
    train()
