import torch
from typing import List
from lenet import LeNet
from torch import nn


def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


if __name__ == "__main__":
    # Reset since we are using a different backend.
    torch._dynamo.reset()

    DEVICE = "cuda"

    #get the dynamo FX graph

    model = LeNet()
    model.to(torch.float32).to(DEVICE)
    optimized_model = torch.compile(model, backend = custom_backend)

    input_shape=(1024, 1, 32, 32)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(torch.float32).to(DEVICE)
    features = optimized_model(input_data)

    #get torch inductor graph
    torch._inductor.config.trace.enabled = True
    optimized_model = torch.compile(model)
    features = optimized_model(input_data)

    #get aotgrad 
    torch._dynamo.reset()
    torch._inductor.config.trace.enabled = False
    optimized_model = torch.compile(model, backend = "aot_eager")
    features = optimized_model(input_data)
    criterion = nn.MSELoss()
    label = torch.ones_like(features)
    loss = criterion(features, label) 
    loss.backward() 

