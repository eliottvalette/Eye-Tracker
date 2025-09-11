import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

        # Encoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool_1 = nn.MaxPool2d(2)

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool_2 = nn.MaxPool2d(2)

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool_3 = nn.MaxPool2d(2)

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool_4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Decoder (transpose conv + convs after skip concat)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up_1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.up_0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder_0 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Heatmap head and soft-argmax to (x,y) in [0,1]
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, return_heatmap=False):
        # Encoder
        e1 = self.encoder_1(x)            # H x W
        p1 = self.pool_1(e1)              # H/2
        e2 = self.encoder_2(p1)           # H/2
        p2 = self.pool_2(e2)              # H/4
        e3 = self.encoder_3(p2)           # H/4
        p3 = self.pool_3(e3)              # H/8
        e4 = self.encoder_4(p3)           # H/8
        p4 = self.pool_4(e4)              # H/16

        b  = self.bottleneck(p4)          # H/16

        # Decoder with skips
        u3 = self.up_3(b)                 # H/8
        d3 = self.decoder_3(torch.cat([u3, e4], dim=1))

        u2 = self.up_2(d3)                # H/4
        d2 = self.decoder_2(torch.cat([u2, e3], dim=1))

        u1 = self.up_1(d2)                # H/2
        d1 = self.decoder_1(torch.cat([u1, e2], dim=1))

        u0 = self.up_0(d1)                # H
        d0 = self.decoder_0(torch.cat([u0, e1], dim=1))

        heatmap = self.out_conv(d0)       # (N,1,H,W)
        print('heatmap', heatmap)
        coords = torch.argmax(heatmap)  # (N,2) in [0,1]

        return (coords, heatmap) if return_heatmap else coords

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))