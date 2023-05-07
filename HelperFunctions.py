import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import sigmoid
from torch.nn import functional
import numpy as np
# from segmentation_models_pytorch.losses import DiceLoss
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    """
    Source:https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, predicted_mask, mask, smooth=1e-10):
        """
        :param predicted_mask:
        :param mask:
        :param smooth:
        :return:
        """
        predicted_mask = sigmoid(predicted_mask).view(-1)
        mask = sigmoid(mask).view(-1)

        intersection = (predicted_mask * mask).sum()
        dice = ((2. * intersection) + smooth) / (predicted_mask.sum() + mask.sum() + smooth)
        return 1 - dice


def pixel_accuracy(output, mask):
    with torch.no_grad():
        # output = torch.argmax(functional.softmax(output, dim=1), dim=1)

        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(predicted_masks, mask, smooth=1e-10, n_classes=11):
    with torch.no_grad():
        predicted_masks = predicted_masks.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = predicted_masks == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def run_epoch(model, classification_loss_function, dataloader, optimizer, train=False, device="cpu",
              loss_weights=[1, 1], logger=None):
    if train:
        model.train()
    else:
        model.eval()

    total_correct_predictions = 0
    total_classification_loss = 0
    total_iou = 0
    total_dice_loss = 0
    total_dice_score = 0
    accuracy = 0
    total_loss = 0
    dataset_samples = len(dataloader.dataset)
    batches_in_dataset = len(dataloader)
    compute_dice_loss = DiceLoss()
    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        print(images.max(), images.min())
        if train:
            optimizer.zero_grad()

        predicted_masks = model(images)

        predicted_masks = torch.argmax(functional.softmax(predicted_masks, dim=1), dim=1).float()
        print(predicted_masks.size())
        plt.imshow(predicted_masks[0, :, :].cpu())
        plt.show()
        # computing classification loss at each pixel
        classification_loss = classification_loss_function(predicted_masks, masks)

        dice_loss = compute_dice_loss(predicted_masks, masks)
        loss = loss_weights[0] * classification_loss + loss_weights[1] * dice_loss
        loss.requires_grad = True

        if train:
            loss.backward()
            optimizer.step()

        if logger is not None:
            logger.log({"Classification loss(batch_wise)": classification_loss, "Dice loss(batch_wise)": dice_loss,
                        "combined loss": loss})

        total_classification_loss += classification_loss.item()
        total_dice_loss += dice_loss
        total_dice_score += (1 - dice_loss)
        accuracy += pixel_accuracy(predicted_masks, masks)
        total_iou += mIoU(predicted_masks, masks)
        total_loss += loss

    loss = total_loss / dataset_samples
    accuracy = accuracy / dataset_samples
    classification_loss = total_classification_loss / dataset_samples
    dice_loss = total_dice_loss / dataset_samples
    dice_score = total_dice_score / dataset_samples
    iou = total_iou / dataset_samples

    return loss, accuracy, classification_loss, dice_loss, dice_score, iou
