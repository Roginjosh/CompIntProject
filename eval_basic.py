# made by following the tutorial at https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import json
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from myClasses import CustomImageDataset, Net

transform = transforms.Compose(
    [ # Going to have to resize for ResNet anyways, and its much faster
        transforms.Resize((224, 224)), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalizing data to make it converge faster
    ]
)

testset = CustomImageDataset( # load test set from csv
    annotations_file="./birds/test.csv", img_dir="./birds", transform=transform
)
label_to_index = {}

# VERY IMPORTANT: this makes sure that the indexes for each class stay constant between training and testing
with open("label_to_index_basic.json", "r") as f:
    label_to_index = json.load(f)

testset.label_to_index = label_to_index # sets class dictionary inside of test set

batch_size = 16
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True, num_workers=0
)


net = Net()
PATH = "./bird_net.pth" # this is where the basic model weights are stored
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0

net.eval() # not necessary since I am not doing any fancy tricks, but good practice

with torch.no_grad():
    for data in tqdm(testloader, desc="Evaluating"):
        images, labels = data # grab a bunch of images and their labels
        outputs = net(images) # throw those images thru the net
        _, predicted = torch.max(outputs, 1) # grab the top prediction for each image
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # compare prediction to ground truth

print(f"Model accuracy on {total} test images: {100 * correct // total}%")
