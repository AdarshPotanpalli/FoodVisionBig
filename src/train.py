"""
This file implements the complete training workflow,
data loading, model building, training, saving the model
"""

from data_setup import create_dataloaders
from model_builder import create_effnetb2_model
from engine import train
from torchvision import datasets, transforms
import torchvision
from pathlib import Path
import torch
from torch import nn
from utils import save_model

#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Hyperparams
EPOCHS = 10

#directory for storing data
data_dir = Path("data")

# Creating effnetb2 model for food 101 classification
effnetb2_food101, effnetb2_transform = create_effnetb2_model(num_classes=101)

#transform for train dataset
effnetb2_transform_augmented = transforms.Compose([
    transforms.TrivialAugmentWide(),
    effnetb2_transform
])

# creating training and testing dataloaders, getting the class names
train_dataloader, test_dataloader, class_names = create_dataloaders(data_dir= data_dir,
                                                                    effnetb2_transform_augmented= effnetb2_transform_augmented,
                                                                    effnetb2_transform= effnetb2_transform,
                                                                    download = False) # keep it true if you want to download dataset



# optimizer
optimizer = torch.optim.Adam(params= effnetb2_food101.parameters(),
                             lr= 0.001)

# loss function
loss_fn = nn.CrossEntropyLoss(label_smoothing= 0.1)

#set seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# train
effnetb2_food101_training_results = train(model= effnetb2_food101,
                                          train_dataloader= train_dataloader,
                                          test_dataloader= test_dataloader,
                                          optimizer = optimizer,
                                          loss_fn= loss_fn,
                                          epochs= EPOCHS,
                                          device = device)

# saving the model
save_model(model = effnetb2_food101,
          target_dir= "models",
          model_name= "09_finetuned_effnetb2_20_percent_food101.pth")
