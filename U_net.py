from DataSet import Dataset
import torch
from torch.utils.data import DataLoader
import os
import time
from HelperFunctions import run_epoch
import segmentation_models_pytorch as segmentation_models
import albumentations as Augmentations
import pandas as pd
import wandb

# These params can be updated to allow for better control of the program(i.e. the control knobs of this code)
run_training = True  # False to run inferences, otherwise it'll start train the model
resume_training = False  # If training needs to be resumed from some epoch
load_model = True  # If you want to load a model previously trained
run_test_set = True  # True to run test set post training

model_name = os.path.basename(__file__).split(".")[
    0]  # Name of the .py file running to standardize the names of the saved files and ease of later use
batch_size = 32
learning_rate = 0.001
pretrained = False  # This option is not valid for Custom model, no pretrained weights exist
epochs = 10
classes = 11
device_name = torch.cuda.get_device_name(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
starting_time = time.time()
class_names = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian",
               "Bicyclist"]

# Path to the dataset
test_images_path = "./Dataset/images_prepped_test"
test_annotation_path = "./Dataset/annotations_prepped_test"

train_images_path = "./Dataset/images_prepped_train"
train_annotation_path = "./Dataset/annotations_prepped_train"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

augmentations = Augmentations.Compose([Augmentations.HorizontalFlip(),
                                       Augmentations.GaussNoise(), Augmentations.Resize(224, 224)])

test_set = Dataset(test_images_path, test_annotation_path, mean, std, transform=augmentations)
train_set = Dataset(train_images_path, train_annotation_path, mean, std, transform=augmentations)

#
train_size = int(0.8 * len(train_set))
validation_size = len(train_set) - train_size
train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, validation_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test_set, batch_size=batch_size,
                      shuffle=False)  # Shuffle = False to create a sequence of frames for output as video

model = segmentation_models.Unet(encoder_name="mobilenet_v2", encoder_weights='imagenet', in_channels=3, classes=11,
                                 activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
wandb.init(
    # set the wandb project where this run will be logged
    project="CV_Assignment_testing",

    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": "U_Net",
        "dataset": "Streets",
        "epochs": epochs,
    }
)

if run_training:
    train_loss_list = []
    train_accuracy_list = []
    train_classification_loss_list = []
    train_dice_loss_list = []
    train_dice_score_list = []
    train_accuracy_list = []
    train_iou_list = []

    validation_loss_list = []
    validation_accuracy_list = []
    validation_classification_loss_list = []
    validation_dice_loss_list = []
    validation_dice_score_list = []
    validation_accuracy_list = []
    validation_iou_list = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss, train_accuracy, train_classification_loss, train_dice_loss, train_dice_score, train_iou = run_epoch(
            model, loss_function, train_loader, optimizer, train=True, device=device,
            loss_weights=[1, 1], logger=wandb)
        wandb.log({"training loss": train_loss, "training_accuracy": train_accuracy,
                   "classification_loss": train_classification_loss, "training_dice_loss": train_dice_loss,
                   "training_dice_score": train_dice_score, "Train MIoU": train_iou})
        scheduler.step()