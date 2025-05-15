# made by following the tutorial at https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from myClasses import CustomImageDataset
import time
import json

transform = transforms.Compose(
    [ # Must resize to fit in ResNet
        transforms.Resize((224, 224)),
        # normalize data to assist with convergence
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ]
)

trainset = CustomImageDataset( # load training set from csv
    annotations_file="./birds/train.csv", img_dir="./birds", transform=transform
)

batch_size = 16
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0
)
index_to_label = {v: k for k, v in trainset.label_to_index.items()}
classes = [index_to_label[i] for i in range(len(index_to_label))]

# VERY IMPORTANT: this makes sure that the indexes for each class stay constant between training and testing
with open("label_to_index_resnet.json", "w") as f:
    json.dump(trainset.label_to_index, f)


net = models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 20)  # 20 classes


criterion = nn.CrossEntropyLoss() # using Cross Entropy Algorithm as our Loss Function
optimizer = optim.Adam(net.parameters(), lr=0.001) # Using Adam over SDG to simplify weight initialization

print("Training Has Begun")
start_time = time.time()
net.train() # not necessary, but good practice
for epoch in range(4):
    running_loss = 0.0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}") # pulled from program used in Dr. Stanley's lab

    for inputs, labels in progress_bar:

        optimizer.zero_grad() # clear old gradients
        outputs = net(inputs) # forward pass
        loss = criterion(outputs, labels) # Compute Loss
        loss.backward() # run backpropagation
        optimizer.step() # use gradients to update model parameters

        running_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

print("Finished Training")
end_time = time.time()
elapsed = end_time - start_time
mins = int(elapsed // 60)
secs = int(elapsed % 60)
print(f"Training completed in {mins} min {secs} sec")


PATH = "./bird_resnet_net.pth"
torch.save(net.state_dict(), PATH) # save weights into a .pth file for later use
