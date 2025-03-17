import torch
import segmentation_models_pytorch as smp
import timm


if False:

    def get_model(num_classes=3):
        """Creates a U-Net model with EfficientNet-B3 Noisy Student encoder."""
        model = smp.Unet(
            encoder_name="timm-efficientnet-b3",  # Use TIMM-based EfficientNet-B3
            encoder_weights="noisy-student",  # Load Noisy Student pretraining
            in_channels=1,
            classes=num_classes,  # Number of output classes
            activation=None,  # Change to 'softmax' or 'sigmoid' if needed
        )
        return model


if True:

    def get_model(num_classes):
        model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
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
