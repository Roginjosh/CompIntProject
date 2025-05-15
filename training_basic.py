# made by following the tutorial at https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from myClasses import CustomImageDataset, Net
import time
import json

transform = transforms.Compose(
    [ # Going to have to resize for ResNet anyways, might as well speed up training quite a bit
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalize data to assist with convergence
    ]
)

trainset = CustomImageDataset( # loat training set from csv
    annotations_file="./birds/train.csv", img_dir="./birds", transform=transform
)

batch_size = 16
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0
)
index_to_label = {v: k for k, v in trainset.label_to_index.items()}
classes = [index_to_label[i] for i in range(len(index_to_label))]

# VERY IMPORTANT: this makes sure that the indexes for each class stay constant between training and testing
with open("label_to_index_basic.json", "w") as f:
    json.dump(trainset.label_to_index, f)


net = Net()


criterion = nn.CrossEntropyLoss() # using Cross Entropy Algorithm as our loss function
optimizer = optim.Adam(net.parameters(), lr=0.001) # Using Adam over SGD to make weights easier to initialize

print("Training Has Begun")
start_time = time.time()
net.train() # not necessary, but good practice
for epoch in range(16):
    running_loss = 0.0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}") # pulled this progress bar from some of the code we use in Dr. Stanley's lab

    for inputs, labels in progress_bar:

        optimizer.zero_grad() # clear old gradients
        outputs = net(inputs) # forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # run backpropagation to compute gradient
        optimizer.step() # use gradients to update model parameters

        running_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

print("Finished Training")
end_time = time.time()
elapsed = end_time - start_time
mins = int(elapsed // 60)
secs = int(elapsed % 60)
print(f"Training completed in {mins} min {secs} sec")


PATH = "./bird_net.pth"
torch.save(net.state_dict(), PATH) # save weights into a .pth file for use later
