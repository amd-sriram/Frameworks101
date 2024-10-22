import time
from lenet import LeNet
import torch

if __name__ == "__main__":

    DEVICE = "cuda"

    model = LeNet()
    model.to(DEVICE).eval()

    input_shape=(1024, 1, 32, 32)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(DEVICE)

    for index in range(10):
        start_time = time.time()    
        features = model(input_data)
        end_time = time.time()
        print ("time taken", (end_time - start_time) * 1000, "ms")
