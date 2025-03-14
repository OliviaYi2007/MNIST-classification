import math
import random
import torch
import numpy as np
import wandb
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda


config = {
    "epochs": 5,
    "batch_size": 16,
    "lr": 1e-3,
    "width": 1024 
}

wandb.init(
    project="project_name",
    name=f"{config['width']}",
    config=config
)

config = wandb.config
transform = Compose ([
    ToTensor(),
    Lambda(lambda image: image.view(784))
])

data_train = MNIST(root="./", download= True, train= True, transform=transform)
data_test = MNIST(root="./", download= True, train= False, transform=transform)

from torch import nn, optim
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def to_t(tensor,device=get_default_device()):
    return tensor.to(device)

class MNISTModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, width),
            nn.ReLU(),
            nn.Linear(width, 10)
        )
        #Use loss function for training, set optimizer with Adam algorithm using parameters of the model
        #Use a scheduler for learning rate, move model to default device
        T_max= 60000/config.batch_size*config.epochs 
        self.loss= nn.CrossEntropyLoss()
        self.optimizer= optim.Adam(self.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max)
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
        self.scheduler.step() 
        return loss.item()
        
mnist_model=MNISTModel(config.width)

from torch.utils.data import DataLoader
BATCH_SIZE=config.batch_size
dataloader_train=DataLoader(data_train,batch_size=BATCH_SIZE, shuffle=True)
dataloader_test=DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

from tqdm import tqdm
EPOCHS =config.epochs
for i in range(EPOCHS):
    for inputs, labels in tqdm(dataloader_train,desc=f"FITTING EPOCH {i}"):
        xs, ys = to_t(inputs), to_t(labels)
        loss = mnist_model.fit(xs,ys)
        wandb.log({"train_loss_step": loss, "lr": mnist_model.scheduler.get_last_lr()[0]})
    print(f"EPOCH {i} completed")
    
    #count the number correctly classified (train data)
    correct_train = 0
    for inputs, labels in dataloader_train:
        xs, ys = to_t(inputs), to_t(labels)
        y_pred = mnist_model.predict(xs)
        correct_train += (ys == y_pred).sum().item()
    
    train_acc = correct_train / (len(dataloader_train) * BATCH_SIZE)
    print(f"TRAIN ACCURACY: {train_acc:.4f}")
    wandb.log({"train_accuracy": train_acc, "width_1": config.width_1, "width_2": config.width_2})
    

    correct_test= 0
    for inputs, labels in dataloader_test:
        xs, ys = to_t(inpupts), to_t(labels)
        y_pred = mnist_model.predict(xs)
        correct_test += (ys == y_pred).sum().item()
    #calculate test accuracy
    test_acc = correct_test/ (len(dataloader_test) * BATCH_SIZE)
    print(f"TEST ACCURACY: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc, "width_1": config.width_1, "width_2": config.width_2})

wandb.finish()
