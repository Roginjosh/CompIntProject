# made by following the tutorial at https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from myClasses import CustomImageDataset


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

testset = CustomImageDataset(
    annotations_file="./birds/test.csv", img_dir="./birds", transform=transform
)

with open("label_to_index_resnet.json", "r") as f:
    label_to_index = json.load(f)

testset.label_to_index = label_to_index

batch_size = 16
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=0
)
index_to_label = {v: k for k, v in testset.label_to_index.items()}
classes = [index_to_label[i] for i in range(len(index_to_label))]


net = models.resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 20)

PATH = "./bird_resnet_net.pth"
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in tqdm(testloader, desc="Evaluating"):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the {total} test images: {100 * correct // total} %")
