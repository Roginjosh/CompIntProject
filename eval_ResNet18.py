# made by following the tutorial at https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from myClasses import CustomImageDataset


transform = transforms.Compose(
    [ # must resize to this for ResNet18
        transforms.Resize((224, 224)),
        # for some reason, normalizing to these values is better than the pre-loaded values
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ]
)

testset = CustomImageDataset( # load test set from csv
    annotations_file="./birds/test.csv", img_dir="./birds", transform=transform
)

# VERY IMPORTANT: this makes sure that the indexes for each class stay constant between training and testing
with open("label_to_index_resnet.json", "r") as f:
    label_to_index = json.load(f)

testset.label_to_index = label_to_index # sets class dictionary inside of test set

batch_size = 16
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=0
)
index_to_label = {v: k for k, v in testset.label_to_index.items()}
classes = [index_to_label[i] for i in range(len(index_to_label))]


net = models.resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 20)

PATH = "./bird_resnet_net.pth" # this is where the basic model weights are stored
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0

net.eval() # not necessary since I am not doing any fancy tricks, but good practice

with torch.no_grad():
    for data in tqdm(testloader, desc="Evaluating"):
        images, labels = data # grab a bunch of images and their labels
        outputs = net(images) # throw those items thru the net
        _, predicted = torch.max(outputs, 1) # grab the top prediciton for each image
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # compare prediction to ground truth

print(f"Accuracy of the network on the {total} test images: {100 * correct // total} %")
