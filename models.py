import timm
import torch.nn as nn


class RSNAEfficientnetModel(nn.Module):
    def __init__(self, pretrained=True):
        self.pretrained = pretrained
        super().__init__()

        # Setup the model
        self.model = timm.create_model(
            "efficientnet_b1",
            pretrained=True,
            drop_rate=0.3,
            drop_path_rate=0.2,
            global_pool="avg",
            num_classes=1,
        )
        # Get gradient checkpointing
        self.model.set_grad_checkpointing(enable=True)

        # Set up the first convolutional layer to have stride 1
        self.model.conv_stem.stride = (1, 1)

        classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(256, 1, bias=True),
        )

        self.model.classifier = classifier

    def forward(self, x):
        x = self.model(x).float()
        return x
