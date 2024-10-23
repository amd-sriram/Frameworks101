from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv = nn.Conv2d(1, 32, 3, 1)


    def forward(self, x):
        output = self.conv(x)
        return output