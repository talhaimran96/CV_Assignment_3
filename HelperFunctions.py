import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import sigmoid
from torch.nn import functional
import numpy as np
# from segmentation_models_pytorch.losses import DiceLoss
import matplotlib.pyplot as plt
from torchvision import transforms as transformations
import segmentation_models_pytorch as segmentation_models


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=12):
    """
    REUSED FROM LAB 4- semantic segmentation
    computes the mean IOU for the model
    :param pred_mask:
    :param mask:
    :param smooth:
    :param n_classes:
    :return:
    """
    with torch.no_grad():
        pred_mask = functional.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def run_epoch(model, classification_loss_function, dice_loss_function, dataloader, optimizer, train=False, device="cpu",
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
    precison = 0
    recall = 0
    f1 = 0
    dataset_samples = len(dataloader.dataset)
    batches_in_dataset = len(dataloader)
    # compute_dice_loss = DiceLoss()
    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        if train:
            optimizer.zero_grad()

        predicted_masks = model(images)

        classification_loss = classification_loss_function(predicted_masks, masks)
        dice_loss = dice_loss_function(predicted_masks, masks)

        loss = loss_weights[0] * classification_loss + loss_weights[1] * dice_loss
        # loss.requires_grad = True

        if train:
            loss.backward()
            optimizer.step()

        if logger is not None:
            logger.log({"Classification loss(batch_wise)": classification_loss, "Dice loss(batch_wise)": dice_loss,
                        "combined loss": loss})

        # class num is hardcodded here, since the dataset is not changing for this task
        pred_mask = functional.softmax(predicted_masks, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        tp, fp, fn, tn = segmentation_models.metrics.get_stats(pred_mask, masks, mode='multiclass',
                                                               num_classes=12)

        total_classification_loss += classification_loss.item()
        total_dice_loss += dice_loss
        total_dice_score += (1 - dice_loss)
        accuracy += segmentation_models.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        recall += segmentation_models.metrics.recall(tp, fp, fn, tn, reduction="micro")
        precison += segmentation_models.metrics.precision(tp, fp, fn, tn, reduction="micro")
        f1 += torch.mean(segmentation_models.metrics.f1_score(tp, fp, fn, tn, reduction=None), 0, False)
        total_iou += mIoU(predicted_masks, masks)
        total_loss += loss

    loss = total_loss / batches_in_dataset
    accuracy = accuracy / batches_in_dataset
    classification_loss = total_classification_loss / batches_in_dataset
    dice_loss = total_dice_loss / batches_in_dataset
    dice_score = total_dice_score / batches_in_dataset
    iou = total_iou / batches_in_dataset
    recall = recall / batches_in_dataset
    precison = precison / batches_in_dataset
    f1 = f1 / batches_in_dataset
    return loss, accuracy, classification_loss, dice_loss, dice_score, iou, recall, precison, f1


def display_for_comparison(image, mask, pred_mask, score):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    reverse_norm = transformations.Compose([transformations.Normalize(mean=[0., 0., 0.],
                                                                      std=[1 / 0.229, 1 / 0.224,
                                                                           1 / 0.225]),
                                            transformations.Normalize(mean=[-0.485, -0.456, -0.406],
                                                                      std=[1., 1., 1.]),
                                            ])
    image = reverse_norm(image)
    ax1.imshow(image.numpy().transpose(1, 2, 0))
    ax1.set_title('Picture');

    ax2.imshow(mask)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask)
    ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()
    plt.show()


def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                            device='cpu'):
    model.eval()
    t = transformations.Compose([transformations.ToTensor(), transformations.Normalize(mean, std)])
    # image = t(image.numpy())
    model.to(device);
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def plot_metrics(model_name, backbone, event, loss, accuracy, classification_loss, dice_loss, dice_score, iou, recall,
                 precision, f1):
    plt.figure(figsize=(12, 8))
    plt.suptitle(model_name + " " + backbone + " " + event)
    plt.subplot(1, 3, 1)
    plt.plot(accuracy, label=f"{event}_accuracy")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(loss, label=f"{event}_loss")
    plt.plot(classification_loss, label=f"{event}_classification_loss")
    plt.plot(dice_loss, label=f"{event}_dice_loss")
    plt.plot(dice_score, label=f"{event}_dice_score")
    plt.plot(iou, label=f"{event}_iou")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(recall, label=f"{event}_recall")
    plt.plot(precision, label=f"{event}_precision")
    plt.legend()

    # plt.subplot(2, 2, 4)
    # plt.plot(f1, label=f"{event}_f1")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"./Results/{model_name}_{backbone}_{event}_Metrics.png")
