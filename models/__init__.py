from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4, resnet20x4
from .resnet import resnet8x4_double
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_0_5
from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101, wide_resnet50_2, resnext50_32x4d, wide_resnet34_4, wide_resnet18_2, wide_resnet34_2
from .shuffleNetv2_imagenet import shufflenet_v2_x1_0 as ShuffleNetV2Imagenet, shufflenet_v2_x0_5, shufflenet_v2_x2_0
from .mobilenetv2_imagenet import mobilenet_v2
from .vgg_imagenet import vgg8_bn as vgg8_imagenet, vgg11_bn as vgg11_imagenet, vgg13_bn as vgg13_imagenet

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'ResNet18': resnet18,
    'ResNet18Double': wide_resnet18_2,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'ResNet101': resnet101,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet8x4_double': resnet8x4_double,
    'resnet32x4': resnet32x4,
    'resnet20x4': resnet20x4,
    'resnext50_32x4d': resnext50_32x4d,
    'ResNet34x4': wide_resnet34_4,
    'ResNet34x2': wide_resnet34_2,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_50_2': wide_resnet50_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg13_imagenet': vgg13_imagenet,
    'vgg11_imagenet': vgg11_imagenet,
    'vgg8_imagenet': vgg8_imagenet,
    'MobileNetV2': mobile_half,
    'MobileNetV2_Imagenet': mobilenet_v2,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_0_5': ShuffleV2_0_5,
    'ShuffleV2_Imagenet': ShuffleNetV2Imagenet,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
}
