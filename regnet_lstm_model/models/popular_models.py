import torch.nn as nn

from torchvision import models
from torchvision.models import (
    ResNet18_Weights, 
    MobileNet_V3_Large_Weights, 
    EfficientNet_B2_Weights, 
    DenseNet121_Weights, 
    MNASNet1_3_Weights, 
    RegNet_X_1_6GF_Weights,
    RegNet_Y_3_2GF_Weights,
)

class PopularModels:

    def __init__(self, choice: str, pretrained: bool, freeze: bool, n_out: int) -> None:
        """
        Pretrained models will pick the best weights.

        Choices:
        - mobilenet_v3_large (MobileNet V3 Large)
        - resnet18 (ResNet 18)
        - efficientnet_b2 (EfficientNet B2)
        - densenet121 (DenseNet 121)
        - mnasnet1_3 (MNASNet 1 3)
        - regnet_x_1_6gf (RegNet X 1 6GF)
        - regnet_y_3_2gf (RegNet Y 3 2GF)
        """
        self.model = None

        if (choice == "mobilenet_v3_large"):
            self.model = models.mobilenet_v3_large()
            if (pretrained):
                self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
                self.weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            if (freeze):
                for name, params in self.model.named_parameters():
                    if ("classifier" not in name):
                        params.requires_grad = False
            self.model.classifier = nn.Sequential(
                nn.Linear(960, 1280),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, n_out),
            )
        
        if (choice == "resnet18"):
            self.model = models.resnet18()
            if (pretrained):
                self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.weights = ResNet18_Weights.IMAGENET1K_V1
            if (freeze):
                for name, params in self.model.named_parameters():
                    if ("fc" not in name):
                        params.requires_grad = False
            n_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features=n_features, out_features=n_out),
            )

        if (choice == "efficientnet_b2"):
            self.model = models.efficientnet_b2()
            if (pretrained):
                self.model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
                self.weights = EfficientNet_B2_Weights.IMAGENET1K_V1
            if (freeze):
                for name, params in self.model.named_parameters():
                    if ("classifier" not in name):
                        params.requires_grad = False
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features=1408, out_features=n_out),
            )

        if (choice == "densenet121"):
            self.model = models.densenet121()
            if (pretrained):
                self.model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.weights = DenseNet121_Weights.IMAGENET1K_V1
            if (freeze):
                for name, params in self.model.named_parameters():
                    if ("classifier" not in name):
                        params.requires_grad = False
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features=1024, out_features=n_out),
            )
            
        if (choice == "mnasnet1_3"):
            self.model = models.mnasnet1_3()
            if (pretrained):
                self.model = models.mnasnet1_3(weights=MNASNet1_3_Weights)
                self.weights = MNASNet1_3_Weights
            if (freeze):
                for name, params in self.model.named_parameters():
                    if ("classifier" not in name):
                        params.requires_grad = False
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, n_out),
            )

        if (choice == "regnet_x_1_6gf"):
            self.model = models.regnet_x_1_6gf()
            if (pretrained):
                self.model = models.regnet_x_1_6gf(weights=RegNet_X_1_6GF_Weights.IMAGENET1K_V2)
                self.weights = RegNet_X_1_6GF_Weights.IMAGENET1K_V2
            if (freeze):
                for name, params in self.model.named_parameters():
                    if ("fc" not in name):
                        params.requires_grad = False
            self.model.fc = nn.Sequential(
                nn.Linear(912, n_out),
            )

        if (choice == "regnet_y_3_2gf"):
            self.model = models.regnet_y_3_2gf()
            if (pretrained):
                self.model = models.regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
                self.weights = RegNet_Y_3_2GF_Weights.IMAGENET1K_V2
            if (freeze):
                for name, params in self.model.named_parameters():
                    if ("fc" not in name):
                        params.requires_grad = False
            self.model.fc = nn.Sequential(
                nn.Linear(1512, n_out),
            )

    def get_model(self):
        return self.model
    
    def get_weights(self):
        return self.weights
