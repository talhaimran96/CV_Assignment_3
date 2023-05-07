import segmentation_models_pytorch as segmentation_models
from torch import nn

model = segmentation_models.Unet(encoder_name="vgg16_bn", encoder_weights='imagenet', in_channels=3, classes=11,
                                 activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

print(model)
