import onnx
import torchvision
from PIL import Image
import time
import onnxruntime as ort
import numpy as np
import torch



device = "cuda"
size = 224
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size),
    torchvision.transforms.CenterCrop(size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

# Pre-process the image and convert into a tensor
img = Image.open("../data/apple.jpeg")
x = transform(img).unsqueeze(0)


#run the pytorch model
model = torchvision.models.resnet50(pretrained=True).to(device)
model.eval()
start_time = time.time()

x_gpu = x.to(device)
out = model(x_gpu)
p = torch.nn.functional.softmax(out, dim=1)
score, index = torch.topk(p, 1)
input_category_id = index[0][0].item()
predicted_confidence = score[0][0].item()
end_time = time.time()
print ("pytorch time taken", (end_time - start_time) * 1000, "ms")


print (ort.get_available_providers())
providers = [
    'MIGraphXExecutionProvider'
]
ort_sess = ort.InferenceSession('../data/resnet50.onnx', providers = providers)

start_time = time.time()
outputs = ort_sess.run(None, {'input.1': x.numpy()})
# Print Result
predicted = outputs[0][0].argmax(0)
end_time = time.time()
print ("onnx time taken", (end_time - start_time) * 1000, "ms")