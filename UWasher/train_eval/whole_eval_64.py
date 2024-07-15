import torch
from UWasher.model.SPPUnet import UWasher
from torchvision.models import resnet18
from thop import profile
from thop import clever_format

if __name__ == "__main__":
    model = UWasher()
    batch_acc = torch.randn(1, 3, 64)
    batch_gyr = torch.randn(1, 3, 64)

    flops, params  = profile(model, inputs = ((batch_acc, batch_gyr) ,))
    print("Flops:", flops / 10e6)
    print("Params:", params / 10e6)
    flops, params = clever_format([flops, params], "%.3f")

    model = resnet18()
    batch_x = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(batch_x,))
    print("Flops:", flops / 10e6)
    print("Params:", params / 10e6)
    flops, params = clever_format([flops, params], "%.3f")