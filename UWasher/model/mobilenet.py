import torch
import torch.nn as nn
import torchvision
from torchvision.models import MobileNetV2


class DSMobileNet(nn.Module):
    def __init__(self, model_name):
        super(DSMobileNet, self).__init__()
        self.model_name = model_name

        if self.model_name == 'mobilenetv2':
            self.backbone1 = MobileNetV2(num_classes=10)
            self.backbone2 = MobileNetV2(num_classes=10)
        elif self.model_name == 'mobilenetv3_small':
            self.backbone1 = torchvision.models.mobilenet_v3_small(pretrained=False, num_classes=10)
            self.backbone2 = torchvision.models.mobilenet_v3_small(pretrained=False, num_classes=10)
        elif self.model_name == 'mobilenetv3_large':
            self.backbone1 = torchvision.models.mobilenet_v3_large(pretrained=False, width_mult=1.0, reduced_tail=False,
                                                                   dilated=False, num_classes=10)
            self.backbone2 = torchvision.models.mobilenet_v3_large(pretrained=False, width_mult=1.0, reduced_tail=False,
                                                                   dilated=False, num_classes=10)

    def forward(self, acc, gyr):
        # acc, gyr = x
        return (self.backbone1(acc) + self.backbone2(gyr)) / 2

class MobileNet(nn.Module):
    def __init__(self, model_name):
        super(MobileNet, self).__init__()
        self.model_name = model_name

        if self.model_name == 'mobilenetv2':
            self.backbone = MobileNetV2(num_classes=10)
        elif self.model_name == 'mobilenetv3_small':
            self.backbone = torchvision.models.mobilenet_v3_small(pretrained=False, num_classes=10)
        elif self.model_name == 'mobilenetv3_large':
            self.backbone = torchvision.models.mobilenet_v3_large(pretrained=False, width_mult=1.0, reduced_tail=False,
                                                                   dilated=False, num_classes=10)

    def forward(self, x):
        acc, gyr = x
        x = acc + gyr
        return self.backbone(x)