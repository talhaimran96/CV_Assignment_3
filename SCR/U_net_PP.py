from DataSet import DataSet
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import re
import time
from HelperFunctions import run_epoch, display_for_comparison, predict_image_mask_miou, plot_metrics
import segmentation_models_pytorch as segmentation_models
from segmentation_models_pytorch.losses import DiceLoss
import albumentations as Augmentations
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from torchvision import transforms as transformations
import cv2

# These params can be updated to allow for better control of the program(i.e. the control knobs of this code)
run_training = True  # False to run inferences, otherwise it'll start train the model
resume_training = False  # If training needs to be resumed from some epoch
load_model = True  # If you want to load a model previously trained
run_test_set = True  # True to run test set post training
generate_video_frames = False  # generates video frames
generate_video = False # To create a video from the frames saved

backbone = 'efficientnet-b4'  # 'resnext50_32x4d'  # 'efficientnet-b4'  # 'resnet101'
model_name = os.path.basename(__file__).split(".")[
    0]  # Name of the .py file running to standardize the names of the saved files and ease of later use
batch_size = 8
learning_rate = 0.001
pretrained = False  # This option is not valid for Custom model, no pretrained weights exist
epochs = 100
classes = 12  # 11 +1 for background
device_name = torch.cuda.get_device_name(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
starting_time = time.time()
class_names = ["Sky", "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol", "Fence", "Car", "Pedestrian",
               "Bicyclist"]

# Path to the dataset
test_images_path = "../Dataset/images_prepped_test"
test_annotation_path = "../Dataset/annotations_prepped_test"

train_images_path = "../Dataset/images_prepped_train"
train_annotation_path = "../Dataset/annotations_prepped_train"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# setting up augmentations
augmentations = Augmentations.Compose(
    [Augmentations.HorizontalFlip(always_apply=False, p=0.5), Augmentations.VerticalFlip(always_apply=False, p=0.2),
     Augmentations.RandomBrightness(always_apply=False, p=0.2),
     Augmentations.GaussianBlur(always_apply=False, p=0.2),
     Augmentations.Cutout(num_holes=5, max_h_size=5, max_w_size=5, always_apply=False, p=0.2),
     Augmentations.GaussNoise(always_apply=False, p=0.2), Augmentations.Resize(256, 256)])

test_set = DataSet(test_images_path, test_annotation_path, mean, std, transform=augmentations)
train_set = DataSet(train_images_path, train_annotation_path, mean, std, transform=augmentations)

# 80-20 split
train_size = int(0.8 * len(train_set))
validation_size = len(train_set) - train_size
train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, validation_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=True)
# model and backbone initialization
model = segmentation_models.UnetPlusPlus(encoder_name=backbone, encoder_weights='imagenet', in_channels=3, classes=12,
                                         activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16],
                                         decoder_attention_type="scse")

classification_loss_function = torch.nn.CrossEntropyLoss()
dice_loss_function = DiceLoss(mode="multiclass")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

model_path = "../Models/"

# Wandb data logger


if run_training:
    train_loss_list = []
    train_accuracy_list = []
    train_classification_loss_list = []
    train_dice_loss_list = []
    train_dice_score_list = []
    train_iou_list = []
    train_recall_list = []
    train_precision_list = []
    train_f1_list = []

    validation_loss_list = []
    validation_accuracy_list = []
    validation_classification_loss_list = []
    validation_dice_loss_list = []
    validation_dice_score_list = []
    validation_iou_list = []
    validation_recall_list = []
    validation_precision_list = []
    validation_f1_list = []

    max_validation_dice = -1
    model.to(device)  # Move the model to device

    wandb.init(
        # set the wandb project where this run will be logged
        project="CV_Assignment_3",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": f"{model_name}_{backbone}",
            "dataset": "Streets",
            "epochs": epochs,
            "Batch_size": batch_size,
        }
    )

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss, train_accuracy, train_classification_loss, train_dice_loss, train_dice_score, train_iou, train_recall, train_precision, train_f1 = run_epoch(
            model, classification_loss_function, dice_loss_function, train_loader, optimizer, train=True, device=device,
            loss_weights=[1, 1], logger=wandb)

        # appending train data to list for plotting later
        train_loss_list.append(train_loss.data.cpu().numpy())
        train_accuracy_list.append(train_accuracy)
        train_classification_loss_list.append(train_classification_loss)
        train_dice_loss_list.append(train_dice_loss.data.cpu().numpy())
        train_dice_score_list.append(train_dice_score.data.cpu().numpy())
        train_iou_list.append(train_iou)
        train_recall_list.append(train_recall)
        train_precision_list.append(train_precision.data.cpu().numpy())
        train_f1_list.append(train_f1.squeeze_(-1).data.cpu().numpy())

        wandb.log({"training loss": train_loss, "training_accuracy": train_accuracy,
                   "training_classification_loss": train_classification_loss, "training_dice_loss": train_dice_loss,
                   "training_dice_score": train_dice_score, "Train MIoU": train_iou,
                   "learning rate ": optimizer.param_groups[0]['lr']})

        print(f"Validation : {epoch}")
        with torch.no_grad():
            validation_loss, validation_accuracy, validation_classification_loss, validation_dice_loss, validation_dice_score, validation_iou, validation_recall, validation_precision, validation_f1 = run_epoch(
                model, classification_loss_function, dice_loss_function, validation_loader, optimizer, train=False,
                device=device,
                loss_weights=[1, 1], logger=None)

            # appending validation data to list for plotting later
            validation_loss_list.append(validation_loss.data.cpu().numpy())
            validation_accuracy_list.append(validation_accuracy)
            validation_classification_loss_list.append(validation_classification_loss)
            validation_dice_loss_list.append(validation_dice_loss.data.cpu().numpy())
            validation_dice_score_list.append(validation_dice_score.data.cpu().numpy())
            validation_iou_list.append(validation_iou)
            validation_recall_list.append(validation_recall)
            validation_precision_list.append(validation_precision.data.cpu().numpy())
            validation_f1_list.append(validation_f1.data.cpu().numpy())

            wandb.log({"validation loss": validation_loss, "validation_accuracy": validation_accuracy,
                       "validation_classification_loss": validation_classification_loss,
                       "validation_dice_loss": validation_dice_loss,
                       "validation_dice_score": validation_dice_score, "validation MIoU": validation_iou})

        scheduler.step()
        if validation_dice_score > max_validation_dice:
            max_validation_dice = validation_dice_score
            print("Max Validation Dice Score Achieved %.2f. Saving model.\n\n" % (max_validation_dice))

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'trianed_epochs': epoch,
                'train_losses': train_loss_list,
                'train_accuracies': train_accuracy_list,
                'train_mIoUs': train_iou,
                'val_losses': validation_loss_list,
                'val_accuracies': validation_loss_list,
                'max_validation_dice': max_validation_dice,
                'val_mIoUs': validation_iou_list,
                'lr': optimizer.param_groups[0]['lr']
            }
            torch.save(checkpoint, model_path + model_name + "_" + backbone + "_" + "_best_dice.pth")

        else:
            print("Max Validation Dice Score did not increase from %.2f\n\n" % (max_validation_dice))

            # Save checkpoint for the last epoch
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'trianed_epochs': epoch,
            'train_losses': train_loss_list,
            'train_accuracies': train_accuracy_list,
            'train_mIoUs': train_iou,
            'val_losses': validation_loss_list,
            'val_accuracies': validation_loss_list,
            'max_validation_dice': max_validation_dice,
            'val_mIoUs': validation_iou_list,
            'lr': optimizer.param_groups[0]['lr']
        }

        torch.save(checkpoint, model_path + model_name + "_" + backbone + "_" + "_check_points.pth")
    wandb.finish()
    execution_time = time.time() - starting_time
    output_report_data = [{"Model/Backbone": model_name, "Start Learning_rate": learning_rate,
                           "End Learning_rate": optimizer.param_groups[0]['lr'], "Batch_size": batch_size,
                           "Training_set_size": train_loader.dataset.__len__(), "Epochs": epochs,
                           "Test_set_size": test_loader.dataset.__len__(),
                           "Validation_set_size": validation_loader.dataset.__len__(), "Tranfer_learning": pretrained,
                           "max_validation_dice": max_validation_dice,
                           "Classification_loss": validation_classification_loss,
                           "Execution_time": execution_time, "Per Epoch Time": execution_time / epochs}]

    output_dataframe = pd.DataFrame(output_report_data)
    output_dataframe.to_csv("../Results/" + model_name + "_" + backbone + "_" + "_experiment_data.csv")

    plot_metrics(model_name, backbone, "Training", train_loss_list,
                 train_accuracy_list,
                 train_classification_loss_list,
                 train_dice_loss_list,
                 train_dice_score_list,
                 train_iou_list,
                 train_recall_list,
                 train_precision_list,
                 train_f1_list)

    plot_metrics(model_name, backbone, "Validation", train_loss_list,
                 validation_accuracy_list,
                 validation_classification_loss_list,
                 validation_dice_loss_list,
                 validation_dice_score_list,
                 validation_iou_list,
                 validation_recall_list,
                 validation_precision_list,
                 validation_f1_list)
    plt.savefig(f"../Results/{model_name}_{backbone}_Metrics.png")
if run_test_set:
    if load_model:
        # 'model': senet.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        # 'trianed_epochs': epoch,
        # 'train_losses': train_loss_list,
        # 'train_accuracies': train_accuracy_list,
        # 'val_losses': val_loss_list,
        # 'val_accuracies': val_accuracy_list,
        # 'classification_loss': val_classification_loss_list,
        # "arousal loss": val_arousal_loss_list,
        # 'valence loss': val_valence_loss_list,
        # 'lr': learning_rate
        checkpoint = torch.load(model_path + model_name + "_" + backbone + "_" + "_best_dice.pth")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # epoch.load_state_dict()
        # epoch
        # train_loss_list
        # train_accuracy_list
        # val_loss_list
        # val_accuracy_list
        # val_classification_loss_list
        # val_arousal_loss_list
        # val_valence_loss_list
        # learning_rate

        model.eval()
        model.to(device)
        with torch.no_grad():
            test_loss, test_accuracy, test_classification_loss, test_dice_loss, test_dice_score, test_iou, test_recall, test_precision, test_f1 = run_epoch(
                model, classification_loss_function, dice_loss_function, test_loader, optimizer, train=False,
                device=device,
                loss_weights=[1, 1], logger=None)

            output_report_data = [{
                "test_loss": test_loss, "test_accuracy": test_accuracy,
                "test_classification_loss": test_classification_loss, "test_dice_loss": test_dice_loss,
                "test_dice_score": test_dice_score, "test_iou": test_iou, "test_recall": test_recall,
                "test_precision": test_precision, "test_f1": test_f1}]

            output_dataframe = pd.DataFrame(output_report_data)
            output_dataframe.to_csv("../Results/" + model_name + "_" + backbone + "_" + "_quantitative_results.csv")

        ## Displaying examples from test dataset
        _, display_set = torch.utils.data.random_split(test_set, [len(test_set) - 4, 4])

        # Reverse normalization for color restoration
        reverse_norm = transformations.Compose([transformations.Normalize(mean=[0., 0., 0.],
                                                                          std=[1 / 0.229, 1 / 0.224,
                                                                               1 / 0.225]),
                                                transformations.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                          std=[1., 1., 1.]),
                                                ])

        plt.figure(figsize=(12, 12))

        for sample in range(len(display_set)):
            image, mask = display_set[sample]
            pred_mask, score = predict_image_mask_miou(model, image, mask)

            image = (reverse_norm(image)).cpu().numpy().transpose(1, 2, 0)
            mask = mask.cpu().numpy()

            plt.subplot(4, 5, ((sample * 5) + 1))
            plt.title("Image")
            plt.imshow(image)
            plt.subplot(4, 5, (sample * 5) + 2)
            plt.title("Ground truth")
            plt.imshow(mask, cmap='viridis')
            plt.subplot(4, 5, (sample * 5) + 3)
            plt.title("Image + Ground truth")
            plt.imshow(image)
            plt.imshow(mask, alpha=0.3, cmap='viridis')
            plt.subplot(4, 5, (sample * 5) + 4)
            plt.title("Predicted Mask")
            plt.imshow(pred_mask)
            plt.subplot(4, 5, (sample * 5) + 5)
            plt.title("Image + Predicted Mask")
            plt.imshow(image)
            plt.imshow(pred_mask, alpha=0.3, cmap='viridis')
        plt.suptitle(model_name + " " + backbone + "_Examples")
        plt.tight_layout()
        plt.savefig(f"../Results/" + model_name + "_" + f"{backbone}_qualitative_results.jpg")
        plt.show()

        augmentations = Augmentations.Compose(
            [Augmentations.Resize(256, 256)])
        plt.figure(figsize=(12, 6))

        if generate_video_frames:
            test_set = DataSet(test_images_path, test_annotation_path, mean, std, transform=augmentations)

            for index in range(len(test_set)):
                image, mask = test_set[index]
                pred_mask, score = predict_image_mask_miou(model, image, mask)

                image = (reverse_norm(image)).cpu().numpy().transpose(1, 2, 0)
                mask = mask.cpu().numpy()

                plt.subplot(1, 5, 1)
                plt.title("Image")
                plt.imshow(image)
                plt.subplot(1, 5, 2)
                plt.title("Ground truth")
                plt.imshow(mask, cmap='viridis')
                plt.subplot(1, 5, 3)
                plt.title("Image + Ground truth")
                plt.imshow(image)
                plt.imshow(mask, alpha=0.3, cmap='viridis')
                plt.subplot(1, 5, 4)
                plt.title("Predicted Mask")
                plt.imshow(pred_mask)
                plt.subplot(1, 5, 5)
                plt.imshow(image)
                plt.title("Image + Predicted Mask")
                plt.imshow(pred_mask, alpha=0.3, cmap='viridis')
                plt.suptitle(model_name + " " + backbone + "_results")
                plt.tight_layout()
                plt.savefig(f"../Results/Video_frames/Unetpp/{index}.jpg")

            if generate_video:
                image_folder = '../Results/Video_frames/Unetpp'
                video_name = f'../Results/Videos/{model_name}_{backbone}_video.avi'
                frames = os.listdir(image_folder)
                frames.sort(key=lambda f: int(re.sub('\D', '', f)))
                # print(frames)
                images = [img for img in frames if img.endswith(".jpg")]
                frame = cv2.imread(os.path.join(image_folder, images[0]))
                height, width, layers = frame.shape

                video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 5, (width, height))

                for image in images:
                    video.write(cv2.imread(os.path.join(image_folder, image)))

                cv2.destroyAllWindows()
                video.release()
