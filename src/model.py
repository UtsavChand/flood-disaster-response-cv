import segmentation_models_pytorch as smp


def build_model(name="unet", pretrained=True):
    weights = "imagenet" if pretrained else None
    if name == "unet":
        return smp.Unet("resnet34", encoder_weights=weights, in_channels=3, classes=1)
    if name == "deeplab":
        return smp.DeepLabV3Plus("resnet34", encoder_weights=weights, in_channels=3, classes=1)
    raise ValueError(f"Unknown model: {name}")