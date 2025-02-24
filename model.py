import torch
import segmentation_models_pytorch as smp


def get_model(num_classes):
    model = smp.Unet(
        encoder_name="efficientnet-b3",  # Encoder architecture: EfficientNet-B3.
        encoder_weights="imagenet",  # Pre-trained on ImageNet.
        in_channels=1,  # Input channels (1 for grayscale).
        classes=num_classes,  # Number of segmentation classes.
    )
    return model


if __name__ == "__main__":
    num_classes = 3
    model = get_model(num_classes)
    model.eval()
    x = torch.randn(1, 1, 160, 160)  # Dummy input for grayscale image.
    with torch.no_grad():
        out = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
