import torch
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

class EyeBranch(nn.Module):
    # Compact CNN for 13x17 eye crops
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 13x17 -> 6x8
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    def forward(self, x):
        return self.net(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                  # 112 -> 56
        )

        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)            # 56x56
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)            # 28x28
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)           # 14x14

        self.left_eye = EyeBranch()
        self.right_eye = EyeBranch()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # 128 (stem) + 32 (left) + 32 (right) + 4 (coords) = 196
        self.head = nn.Sequential(
            nn.Linear(196, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, full_img, left_img, right_img, coords):
        # Global context
        x = self.stem(full_img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        global_feat = self.flatten(self.global_pool(x))
        
        # Local eye features
        left_feat = self.left_eye(left_img)
        right_feat = self.right_eye(right_img)
        
        # Concat all + coords
        concat_feat = torch.cat([global_feat, left_feat, right_feat, coords], dim=1)
        
        return self.head(concat_feat)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, map_location="cpu", strict: bool = False):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=strict)
        self.eval()