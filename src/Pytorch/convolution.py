import torch
from torch.profiler import profile, record_function, ProfilerActivity
from Pytorch.simple_model import SimpleModel



model = SimpleModel().cuda()
input = torch.rand((1, 1, 28, 28)).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        model(input)
prof.export_chrome_trace("../data/profile.json")