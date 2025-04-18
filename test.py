import torch

model0 = torch.hub.load("./yolov5", "custom", path="./models/line_test_best100.pt", source="local", force_reload=True, device=0)
model1 = torch.hub.load("./yolov5", "custom", path="./models/helm_test_best50.pt", source="local", force_reload=True, device=1)

device0 = next(model0.model.parameters()).device
device1 = next(model1.model.parameters()).device

print(f"Model0 berjalan di: {device0}")
print(f"Model1 berjalan di: {device1}")