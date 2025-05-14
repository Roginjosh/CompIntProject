# made by following the tutorial at https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import json
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from myClasses import CustomImageDataset, Net

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

testset = CustomImageDataset(
    annotations_file="./birds/test.csv", img_dir="./birds", transform=transform
)
label_to_index = {}
with open("label_to_index_basic.json", "r") as f:
    label_to_index = json.load(f)

testset.label_to_index = label_to_index

batch_size = 16
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=0
)


net = Net()
PATH = "./bird_net.pth"
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

print(f"Model accuracy on {total} test images: {100 * correct // total}%")
