import torch.nn as nn


class RSNAEfficientnetModel(nn.Module):
    def __init__(self, pretrained=True):
        self.pretrained = pretrained
        super().__init__()

        # Setup the model
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            drop_rate=0.3,
            drop_path_rate=0.2,
            global_pool="avg",
            num_classes=1,
        )
        # Set up the first convolutional layer to have stride 1
        self.model.conv_stem.stride = (1, 1)

    def forward(self, x):
        x = self.model(x)
        return x
