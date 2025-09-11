import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial gating map per channel in [0,1]
        self.heatmap = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.pass_1 = nn.Sequential( 
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 224x224 -> 224x224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 112x112
        )

        self.pass_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 56x56
        )

        self.pass_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 56x56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 28x28
        )

        self.pass_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 14x14
        )

        # 224x224 input -> 14x14x128 before FC
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        h = self.heatmap(x)         # (N,3,H,W) in [0,1]
        x = x * (1.0 + h)           # gated residual on input
        x = self.pass_1(x)
        x = self.pass_2(x)
        x = self.pass_3(x)
        x = self.pass_4(x)
        x = self.fc(x)
        x = torch.sigmoid(x)        # normalize to [0,1]
        return x
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path, map_location="cpu", strict: bool = False):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=strict)  # strict=False avoids key mismatches across arches
        self.eval()

