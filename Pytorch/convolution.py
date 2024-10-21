from torch import nn
import torch
from torch.profiler import profile, record_function, ProfilerActivity

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv = nn.Conv2d(1, 32, 3, 1)


    def forward(self, x):
        output = self.conv(x)
        return output


model = SimpleModel().cuda()
input = torch.rand((1, 1, 28, 28)).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        model(input)
prof.export_chrome_trace("profile.json")