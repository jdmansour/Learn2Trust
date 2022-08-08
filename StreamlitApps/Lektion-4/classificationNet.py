import torch
import torch.nn as nn

class classificationCNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv_block0 = nn.Sequential(
                nn.Conv2d(1, 10, 3, bias=False),
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Conv2d(10, 10, 3, bias=False),
                nn.BatchNorm2d(10),
                nn.ReLU()
            )

            self.conv_block1 = nn.Sequential(
                nn.Conv2d(10, 20, 3, bias=False),
                nn.BatchNorm2d(20),
                nn.ReLU(),
                nn.Conv2d(20, 20, 3, bias=False),
                nn.BatchNorm2d(20),
                nn.ReLU()
            )

            self.conv_block2 = nn.Sequential(
                nn.Conv2d(20, 40, 3, bias=False),
                nn.BatchNorm2d(40),
                nn.ReLU(),
                nn.Conv2d(40, 40, 3, bias=False),
                nn.BatchNorm2d(40),
                nn.ReLU()
            )

            self.conv_block3 = nn.Sequential(
                nn.Conv2d(40, 80, 3, bias=False),
                nn.BatchNorm2d(80),
                nn.ReLU(),
                nn.Conv2d(80, 80, 3, bias=False),
                nn.BatchNorm2d(80),
                nn.ReLU()
            )


            self.fc_block = nn.Sequential(
                nn.Linear(4*4*80, 120),
                nn.ReLU(),
                nn.Linear(120, 40),
                nn.ReLU(),
                nn.Linear(40, 2)
            )

            self.maxPool = nn.MaxPool2d(2)

        def forward(self, x):

            x = self.maxPool(self.conv_block0(x))
            x = self.maxPool(self.conv_block1(x))
            x = self.maxPool(self.conv_block2(x))
            x = self.maxPool(self.conv_block3(x))
            x = torch.flatten(x,1)
            x = self.fc_block(x)

            return x