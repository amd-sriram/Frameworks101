import torchvision
import numpy as np
from PIL import Image
import torch 

model = torchvision.models.resnet50(pretrained=True)

size= 224
# Pre-process the image and convert into a tensor
transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize(size),
     torchvision.transforms.CenterCrop(size),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
 ])

img = Image.fromarray(np.ones((size, size, 3), dtype=np.uint8))
x = transform(img).unsqueeze(0)
torch.onnx.export(model, x, 'data/resnet50.onnx')

