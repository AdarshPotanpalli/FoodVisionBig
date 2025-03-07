
""" Contains functionality for creating PyTorch DataLoaders for image classification data
"""

from torchvision import datasets, transforms
import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os

# Hyperparams
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

def split_food_101(dataset: torchvision.datasets,
                   split_ratio:int = 0.2,
                   seed:int= 42):
  
  """
  Takes in dataset and returns randomized splits having given ratio of data

  Args:
    dataset: train or test dataset
    split_ratio: ratio of randomized split
    seed: pseudo random generator

  Returns:
    Tuple of randomized splits
  
  """

  length_1 = int(len(dataset)*split_ratio) # 20% length
  length_2 = len(dataset) - length_1 # 80% length

  split_1, split_2 = random_split(dataset= dataset,
                                  lengths= [length_1, length_2],
                                  generator= torch.manual_seed(seed))

  return split_1, split_2

def create_dataloaders(data_dir: str,
                       effnetb2_transform_augmented: transforms,
                       effnetb2_transform: transforms,
                       batch_size: int = BATCH_SIZE,
                       num_workers: int = NUM_WORKERS,
                       download: bool = False):

  """Creates training and testing dataloaders

  takes in data directory and creates train and test datasets then converts
  them into train and test dataloaders

  Args:
    data_dir: path to dataset
    effnetb2_transform_augmented: torchvision transforms to transform images with augmentation
    effnetb2_transform: torchvision transforms to transform images without augmentation
    batch_size: number of samples per batch in dataloaders
    num_workers: number of cpu cores to load the dataset into dataloader
    download: whether to download the dataset or not

  Returns:
    Tuple of (train_dataloader, test_dataloader, class_names)

  Example usage:
    train_dataloader, test_dataloader, class_names = create_dataloaders(
      data_dir = "path/to/data_dir",
      effnetb2_transform_augmented = some_transform,
      effnetb2_transform = some_transform,
      batch_size = 32,
      num_workers = 4
    )

  """

  train_data = datasets.Food101(root= data_dir,
                                split= "train",
                                transform = effnetb2_transform_augmented,
                                download= download)

  test_data = datasets.Food101(root= data_dir,
                              split= "test",
                              transform= effnetb2_transform, # no need to apply augmentation on test data
                              download= download)
  
  train_data_20_percent, _ = split_food_101(train_data, split_ratio=0.2) # 20% of food101
  test_data_20_percent, _ = split_food_101(test_data, split_ratio= 0.2) # 20% of food 101

  class_names = train_data.classes

  # Turn train and test Datasets into DataLoaders
  train_dataloader = DataLoader(dataset= train_data_20_percent,
                                batch_size = batch_size, # how many samples per batch?
                                num_workers= num_workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle = True,
                                pin_memory= True)

  test_dataloader = DataLoader(dataset= test_data_20_percent,
                              batch_size = batch_size,
                              num_workers= num_workers,
                              shuffle = False,
                              pin_memory= True) # don't usually need to shuffle testing data

  return train_dataloader, test_dataloader, class_names
