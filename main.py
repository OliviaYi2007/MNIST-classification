import torchvision
import torch
import numpy as np
import wandb
import math
import random
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda

for _ in range(5):
    wandb.init(
        project="digitrecognition",
        config={
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": random.uniform(0.01, 0.80),
            })

    # Copy your config
    config = wandb.config

#Convert image to a tensor, reshape to vector of size 784
transform = Compose ([
    ToTensor(),
    Lambda(lambda image: image.view(784))
])
data_train = MNIST(root="./", download= True, train= True, transform=transform)
data_test = MNIST(root="./", download= True, train= False, transform=transform)

from torch import nn, optim
#Return the default device (GPU if available, otherwise CPU)
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#Move tensor to specified device
def to_t(tensor,device=get_default_device()):
    return tensor.to(device)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
        self.loss= nn.CrossEntropyLoss()
        self.optimizer= optim.Adam(self.parameters())
        self.to(get_default_device())

    def forward(self,X):
        return self.layers(X)
   
    def predict(self, X):
        with torch.no_grad():
            return torch.argmax(self.forward(X),axis=-1)
        
    def fit(self, X, Y):
        self.optimizer.zero_grad()
        y_pred= self.forward(X)
        loss= self.loss(y_pred,Y)
        loss.backward()
        self.optimizer.step() 
        return loss.item()
    
mnist_model=MNISTModel()

from torch.utils.data import DataLoader

BATCH_SIZE=16
dataloader_train=DataLoader(data_train,batch_size=BATCH_SIZE, shuffle=True)
dataloader_test=DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

from tqdm import tqdm

EPOCHS =5

for i in range(EPOCHS):
    total_loss= 0
    for r_xs, r_ys in tqdm(dataloader_train,desc=f"FITTING EPOCH {i}"):
        xs, ys = to_t(r_xs), to_t(r_ys)
        total_loss += mnist_model.fit(xs,ys)
    total_loss /= len(dataloader_train)
    print(f"EPOCH {i}: {total_loss:.4f}")

correct = 0
for r_xs, r_ys in dataloader_test:
    xs, ys = to_t(r_xs), to_t(r_ys)
    y_pred = mnist_model.predict(xs)
    correct += (ys == y_pred).sum()
acc = correct / (len(dataloader_test) * BATCH_SIZE)
print(f"ACCURACY: {acc:.4f}")
wandb.finish()