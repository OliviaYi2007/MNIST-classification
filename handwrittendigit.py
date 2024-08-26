#Import libraries
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
    "width": 1024 #width can be changed 
}

# Initialize wandb, set run name using widths
wandb.init(
    project="width_effect_on_accuracy",
    name=f"{config['width']}",
    config=config
)
# Copy config into local variable
config = wandb.config

#Convert image to a tensor, reshape to vector of size 784
transform = Compose ([
    ToTensor(),
    Lambda(lambda image: image.view(784))
])
#Load training and testing dataset from MNIST
data_train = MNIST(root="./", download= True, train= True, transform=transform)
data_test = MNIST(root="./", download= True, train= False, transform=transform)

from torch import nn, optim
#Return the default device (GPU if available, otherwise CPU)
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#Move tensor to specified device
def to_t(tensor,device=get_default_device()):
    return tensor.to(device)

#Define neural network model class inheriting from the nn.Module (base class for neural network modules in PyTorch)
class MNISTModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, width),
            nn.ReLU(),
            nn.Linear(width, 10)
        )
        T_max= 60000/config.batch_size*config.epochs #Maximum number of iterations for the cosine annealing schedule.
        self.loss= nn.CrossEntropyLoss()
        self.optimizer= optim.Adam(self.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max)
        self.to(get_default_device())
    #Transforms input X (batch of MNIST images flattened into vectors of size 784) into output through the NN layers
    def forward(self,X):
        return self.layers(X)
    
    #Process input X through model and return predicted class labels
    def predict(self, X):
        #Disable gradient calculation
        with torch.no_grad():
            #return indices of the maximum value of the last dimension (output)
            return torch.argmax(self.forward(X),axis=-1)
    
    def fit(self, X, Y):
        #Reset gradient
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
#change log for loss to every step rather than average
for i in range(EPOCHS):
    for r_xs, r_ys in tqdm(dataloader_train,desc=f"FITTING EPOCH {i}"):
        xs, ys = to_t(r_xs), to_t(r_ys)
        loss = mnist_model.fit(xs,ys)
        wandb.log({"train_loss_step": loss, "lr": mnist_model.scheduler.get_last_lr()[0]})
    print(f"EPOCH {i} completed")
    
#for every EPOCH log training and test accuracy
    correct_train = 0
    for r_xs, r_ys in dataloader_train:
        xs, ys = to_t(r_xs), to_t(r_ys)
        y_pred = mnist_model.predict(xs)
        correct_train += (ys == y_pred).sum().item()
    train_acc = correct_train / (len(dataloader_train) * BATCH_SIZE)
    print(f"TRAIN ACCURACY: {train_acc:.4f}")
    wandb.log({"train_accuracy": train_acc, "width":config.width})

    correct_test= 0
    for r_xs, r_ys in dataloader_test:
        xs, ys = to_t(r_xs), to_t(r_ys)
        y_pred = mnist_model.predict(xs)
        correct_test += (ys == y_pred).sum().item()
    test_acc = correct_test/ (len(dataloader_test) * BATCH_SIZE)
    print(f"TEST ACCURACY: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc, "width":config.width})
wandb.finish()
