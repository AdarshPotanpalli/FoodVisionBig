# Food Vision Big

**EfficientNet B2** is trained on 20% of **FOOD101** for multiclass food-image classification.

---
---
`data_setup.py`:
* `create_dataloaders`: takes in data directory and creates train and test datasets(FOOD101) then converts them into train and test dataloaders
* `split_food_101`: takes in data directory and creates train and test datasets then converts them into train and test dataloaders
---
`engine.py`: 
* `train_step`: Trains a PyTorch model for a single epoch
* `test_step`: Trains a PyTorch model for a single epoch
* `train`: Trains a PyTorch model for a single epoch
---
`model_builder.py`:
* `create_effnetb2_model`: Trains a PyTorch model for a single epoch
---
`train.py`: 
* Implements the complete training workflow, data loading, model building, training, saving the model
---
`utils.py`:
* `save_model`: Saves a PyTorch model to a target directory
---
---
### Inspired from: Pytorch for Deep Learning Bootcamp, Udemy
### Course certificate: [Certificate](https://www.udemy.com/certificate/UC-4837897e-9274-463f-a5ed-cc43aad6269c/)
