## Description
Custom python package that implements some Neural Networks modified for the task of Satellite Imagery classification.

## Functions and Classes
- `UNet` : CLASS - building a U-Net model taking inspiration from https://arxiv.org/abs/1505.04597 but adapting the model to multi-class classification tasks.
- `VGG19UNet` : CLASS - building a U-Net model taking inspiration from https://arxiv.org/abs/1505.04597 that uses a pre-trained VGG19 (https://arxiv.org/abs/1409.1556) as encoder (feature extractor) and adapting it to multi-class classification tasks.
- `ResNet50Unet` : CLASS - building a U-Net model taking inspiration from https://arxiv.org/abs/1505.04597 that uses a pre-trained ResNet50 (https://arxiv.org/abs/1512.03385) as encoder (feature extractor) and adapting it to multi-class classification tasks.

## Tests
- `test_unet` - test the **UNet** class
- `test_vgg19_unet` - test the **VGG19UNet** class
- `test_resnet50_unet` - test the **ResNet50Unet** class
