import torch
import torchvision
from torch import nn
from typing import Tuple

def create_effnetb2_model(
    num_classes:int = 101,
    seed:int = 42
)-> Tuple[torch.nn.Module, torchvision.transforms.Compose]:

  """
  Creates trainable effnetb2 model through transfer learning

  Args:
    num_classes: number of output classes
    seed: pseudo random generator

  Returns:
    Tuple of (effnetb2_model, effnetb2_transform)

  """

  # 1.1 getting the transform suitable for effnetb2
  effnetb2_transform = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()

  # 1.2 creates the model with default weights
  effnetb2_model = torchvision.models.efficientnet_b2(weights = "DEFAULT")

  # 2.1 freezing the network backbone
  for param in effnetb2_model.parameters():
    param.requires_grad = False

  # 2.2 updating the final layer
  torch.manual_seed(seed)
  effnetb2_model.classifier = nn.Sequential(
      nn.Dropout(p= 0.3, inplace = True),
      nn.Linear(in_features= 1408, out_features = num_classes, bias= True)
  )

  # 3. return the model and the transforms
  return effnetb2_model, effnetb2_transform
