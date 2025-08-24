import torch
import torch.nn as nn

class Network1(nn.Module):
    def __init__(self, n_classes: int, img_size: int):
        super().__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
                                     nn.ReLU(),
                                     nn.MaxPool2d(2, 2)])
        
        self.fcs = nn.Sequential(*[nn.Linear(img_size // 2**2 * 16 * 16, 120),
                                   nn.ReLU(),
                                   nn.Linear(120, 80),
                                   nn.ReLU(),
                                   nn.Linear(80, n_classes)])

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1) # flatten all dim except from batch.
        y = self.fcs(x)

        return y
    
class Network2(nn.Module):
    def __init__(self, n_classes: int, img_size: int):
        super().__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool2d(4, 4)])
        
        self.fc = nn.Linear((img_size // 4) * (img_size // 4) * 1, n_classes)

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1) # flatten all dim except from batch.
        y = self.fc(x)

        return y