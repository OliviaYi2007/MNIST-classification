#Import libraries
import math
import random
import torch
import numpy as np
import wandb
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda

#Dictionary that contains hyperparameters for the training process (values can be changed)
config = {
    "epochs": 5,
    "batch_size": 16,
    "lr": 1e-3,
    "width": 1024 
}

# Initialize wandb, project name and run name can be edited
wandb.init(
    project="project_name",
    name=f"{config['width']}",
    config=config
)

# Overwrite local variable with config stored in wandb allowing for potential updates from the wandb interface
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
        #Use loss function for training, set optimizer with Adam algorithm using parameters of the model
        #Use a scheduler for learning rate, move model to default device
        T_max= 60000/config.batch_size*config.epochs #Maximum number of iterations for the cosine annealing schedule.
        self.loss= nn.CrossEntropyLoss()
        self.optimizer= optim.Adam(self.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max)
        self.to(get_default_device())
    
    #Transforms input X (batch of MNIST images flattened into vectors of size 784) into output through the NN layers
    def forward(self,X):
        return self.layers(X)
    
    #Process input X through model and return predicted class labels (index of highest output value)
    def predict(self, X):
        #Disable gradient calculation
        with torch.no_grad():
            #return indices of the maximum value of the last dimension (output)
            return torch.argmax(self.forward(X),axis=-1)
    
    def fit(self, X, Y):
         #Reset gradient
        self.optimizer.zero_grad()
        #compute model prediction
        y_pred= self.forward(X)
        #calculate loss
        loss= self.loss(y_pred,Y)
        #backpropagation
        loss.backward()
        #Updates model parameters, adjust the learning rate
        self.optimizer.step()
        self.scheduler.step() 
        #return loss
        return loss.item()

#Creates an instance of the MNISTModel class with the specified width from the config
mnist_model=MNISTModel(config.width)

from torch.utils.data import DataLoader
#set batch size to batch size in config
BATCH_SIZE=config.batch_size
#Shuffle data
dataloader_train=DataLoader(data_train,batch_size=BATCH_SIZE, shuffle=True)
dataloader_test=DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

from tqdm import tqdm
#set number of epochs to the epochs in config
EPOCHS =config.epochs
#training loop looping for each epoch
#Loops through batches of training data, with a progress bar provided by tqdm
for i in range(EPOCHS):
    for inputs, labels in tqdm(dataloader_train,desc=f"FITTING EPOCH {i}"):
        #move the batch data to the appropriate device 
        xs, ys = to_t(inputs), to_t(labels)
        #calculate loss
        loss = mnist_model.fit(xs,ys)
        #log onto wandb
        wandb.log({"train_loss_step": loss, "lr": mnist_model.scheduler.get_last_lr()[0]})
    print(f"EPOCH {i} completed")
    
    #count the number correctly classified (train data)
    correct_train = 0
    for inputs, labels in dataloader_train:
        xs, ys = to_t(inputs), to_t(labels)
        #prediction of class label
        y_pred = mnist_model.predict(xs)
        #compares prediction with given label, if true, add number to correct_test
        correct_train += (ys == y_pred).sum().item()
    #calculate train accuracy
    train_acc = correct_train / (len(dataloader_train) * BATCH_SIZE)
    print(f"TRAIN ACCURACY: {train_acc:.4f}")
    wandb.log({"train_accuracy": train_acc, "width_1": config.width_1, "width_2": config.width_2})
    
    #count number correctly classified (test data)
    correct_test= 0
    for inputs, labels in dataloader_test:
        xs, ys = to_t(inpupts), to_t(labels)
        #prediction of class label
        y_pred = mnist_model.predict(xs)
        correct_test += (ys == y_pred).sum().item()
    #calculate test accuracy
    test_acc = correct_test/ (len(dataloader_test) * BATCH_SIZE)
    print(f"TEST ACCURACY: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc, "width_1": config.width_1, "width_2": config.width_2})

wandb.finish()
