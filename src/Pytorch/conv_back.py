import torch
from torch.profiler import profile, record_function, ProfilerActivity
from Pytorch.simple_model import SimpleModel
from torch import nn


model = SimpleModel().cuda()
input = torch.rand((1, 1, 28, 28)).cuda()
output = model(input)
label = torch.ones_like(output)
criterion = nn.MSELoss()
loss = criterion(output, label)



with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        loss.backward()

prof.export_chrome_trace("data/profile_back.json")