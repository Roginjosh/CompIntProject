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
    [
        # transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = CustomImageDataset(
    annotations_file="./birds/train.csv", img_dir="./birds", transform=transform
)

batch_size = 16
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0
)
index_to_label = {v: k for k, v in trainset.label_to_index.items()}
classes = [index_to_label[i] for i in range(len(index_to_label))]

with open("label_to_index_basic.json", "w") as f:
    json.dump(trainset.label_to_index, f)


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("Training Has Begun")
start_time = time.time()
net.train()
for epoch in range(16):
    running_loss = 0.0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}")

    for inputs, labels in progress_bar:

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

print("Finished Training")
end_time = time.time()
elapsed = end_time - start_time
mins = int(elapsed // 60)
secs = int(elapsed % 60)
print(f"Training completed in {mins} min {secs} sec")


PATH = "./bird_net.pth"
torch.save(net.state_dict(), PATH)
