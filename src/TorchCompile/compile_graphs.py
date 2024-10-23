import torch
from typing import List
from TorchCompile.model import Model
from torch._functorch.aot_autograd import aot_module_simplified


def print_fx_graph(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


def aot_backend(gm, sample_inputs): 
    # Forward compiler capture
    def fw(gm, sample_inputs):
        print("custom backend called with AOT autograd graph:")
        gm.graph.print_tabular()
        return gm.forward
    
    # Backward compiler capture
    def bw(gm, sample_inputs):
        
        gm.graph.print_tabular()
        return gm.forward
    
    # Call AOTAutograd
    gm_forward = aot_module_simplified(gm,sample_inputs,
                                       fw_compiler=fw,
                                       bw_compiler=bw)

    return gm_forward





if __name__ == "__main__":
    # Reset since we are using a different backend.
    torch._dynamo.reset()

    DEVICE = "cuda"
    model = Model()
    model.to(torch.float32).to(DEVICE)
    input_shape=(1, 10)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(torch.float32).to(DEVICE)

    #get the dynamo FX graph
    optimized_model = torch.compile(model, backend = print_fx_graph)
    features = optimized_model(input_data)
    

    #get torch inductor graph
    torch._inductor.config.trace.enabled = True
    optimized_model = torch.compile(model)
    features = optimized_model(input_data)

    #get aot autograd graph
    torch._dynamo.reset()
    ouput = torch.randn(2).to(DEVICE)
    compiled_f = torch.compile(model, backend=aot_backend)
    loss= torch.nn.functional.mse_loss(compiled_f(input_data), ouput)
    out = loss.backward()
  

   

