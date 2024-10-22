import time
from TorchCompile.model import LeNet
import torch
import torch._dynamo
import warnings


if __name__ == "__main__":


    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True

    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )


    # Reset since we are using a different mode.
    torch._dynamo.reset()

    DEVICE = "cuda"

    model = LeNet()
    model.to(torch.float32).to(DEVICE)
    optimized_model = torch.compile(model)
    
    

    input_shape=(1024, 1, 32, 32)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(torch.float32).to(DEVICE)

    with torch.no_grad():
        for index in range(10):
            start_time = time.time()    
            features = optimized_model(input_data)
            end_time = time.time()
            print ("time taken", (end_time - start_time) * 1000, "ms")
