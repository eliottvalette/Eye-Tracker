import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pass_1 = nn.Sequential( 
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 224x224x3 -> 224x224x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 112x112x16
            nn.Dropout(0.1),
        )

        self.pass_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 112x112x16 -> 112x112x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 56x56x32
            nn.Dropout(0.1),
        )

        self.pass_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 56x56x32 -> 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 28x28x64
            nn.Dropout(0.1),
        )

        self.pass_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 28x28x64 -> 28x28x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # -> 14x14x64
            nn.Dropout(0.1),
        )

        # Removed raw image concatenation to prevent overfitting
        # Using only learned features from convolutional layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, raw_image):
        x = self.pass_1(raw_image)
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


