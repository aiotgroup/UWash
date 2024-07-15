import torch
from ablation_unet import UWasher, AblationUNetConfig
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchvision.models import MobileNetV2
import torchvision

if __name__ == '__main__':
    backbone = UWasher(AblationUNetConfig(True, False, True))
    accData, gyrData = torch.randn(2, 3, 64), torch.randn(2, 3, 64)
    flops = FlopCountAnalysis(backbone, (accData, gyrData))
    print("unet".center(100, "="))
    print(flop_count_table(flops))

    # from UWasher.model.resnet1d import resnet1d, ResNet1DConfig
    # from UWasher.model.mobilenet import MobileNet
    #
    # backbone = resnet1d(ResNet1DConfig())
    # inputData = torch.randn(1, 6, 64)
    # flops = FlopCountAnalysis(backbone, inputData)
    # print("Resnet1D-18".center(100, "="))
    # print(flop_count_table(flops))
    #
    # backbone = MobileNet('mobilenetv3_small')
    # accData, gyrData = torch.randn(1, 3, 64, 1), torch.randn(1, 3, 64, 1)
    # flops = FlopCountAnalysis(backbone, (accData, gyrData))
    # print("MobileNetV3-small".center(100, "="))
    # print(flop_count_table(flops))

    from UWasher.model.utformer import UTFormerEVO, UTFormerEVOConfig

    backbone = UTFormerEVO(UTFormerEVOConfig("utformer_3", 3))
    accData, gyrData = torch.randn(1, 3, 64), torch.randn(1, 3, 64)
    flops = FlopCountAnalysis(backbone, [accData, gyrData])
    print("UTFormer-3".center(100, "="))
    print(flop_count_table(flops))