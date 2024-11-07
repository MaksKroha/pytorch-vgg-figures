import torch
import torchvision
import torch.nn.functional as Func
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Model import CNN

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as pyplt

# Hyperparameters
epochs = 10
batch_size = 512
lr = 0.001
train_device = "cpu"
test_device = "cpu"

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST("dataset", True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST("dataset", False, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, 1, shuffle=True)

model = CNN()
model.to(train_device)
model.train()
criterion = torch.nn.CrossEntropyLoss()
try:
    model_state_dict = torch.load("parameters/model_state_dict.pt", weights_only=True)
    model.load_state_dict(model_state_dict)
    # if there are not studied parameters model starts to train
except (FileNotFoundError, EOFError):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print(epoch)
        for images, labels in train_dataloader:
            images, labels = images.to(train_device), labels.to(train_device)
            logits = model(images)  # forward pass
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            model.zero_grad()  # not required
            # через те що за замовчуванням попередні градієнти додаються до поточних при обчисленні
            # optimizer.zero_grad() обнуляє також і градієнти моделі

            loss.backward()
            optimizer.step()  # update weights
    torch.save(model.state_dict(), "parameters/model_state_dict.pt")
    # model state dict зберігає параметри для моделі (weights, biases)
    # та параметри для batch norm (dispersion, average, batches counter)

model.eval()
model.to(test_device)
torch.set_grad_enabled(False)
x = []
y = []
for idx, (img, label) in enumerate(test_dataloader):
    images, labels = img.to(train_device), label.to(train_device)
    logits = Func.softmax(model.forward(img))
    res = (1.0 - logits[0, label.item()].item()) * 100
    x.append(idx)
    y.append(res)
torch.set_grad_enabled(True)

pyplt.scatter(x, y, label="Похибка в %")
pyplt.ylim(0, 100)
pyplt.xlabel("index")
pyplt.ylabel("loss")
pyplt.show()
# state_dict - словник який по шарам зберігає навчальні параметри
# model.state_dict(), optimizer.state_dict()...
# model.train() - переводить модель в режим тренування (якщо використ. dropout, batchnorm... вкл)
# model.eval() - переводить модель в режим оцінювання (якщо використ. dropout, batchnorm... викл)
# torch.no_grad() - авдключає автоматичне обчислення градієнтів для ефективності (під час тесту)