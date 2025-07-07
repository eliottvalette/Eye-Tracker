import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pass_1 = nn.Sequential( 
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # 224x224x1 -> 224x224x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 224x224x16 -> 112x112x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 112x112x16 -> 112x112x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112x32 -> 56x56x32
        )
        self.pass_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 56x56x32 -> 56x56x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56x64 -> 28x28x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 28x28x64 -> 28x28x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28x128 -> 14x14x128
        )
        # For 224x224 input, the feature map size after pass_2 will be 14x14x128
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 128 * 14),
            nn.ReLU(),
            nn.Dropout(0.03),
            nn.Linear(128 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(128, 2)  # Output x, y coordinates
        )
    
    def forward(self, x):
        x = self.pass_1(x)
        x = self.pass_2(x)
        x = self.fc(x)
        x = torch.sigmoid(x)  # Apply sigmoid to normalize output to [0,1]
        return x
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))