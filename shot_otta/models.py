"""Model definitions compatible with SHOT source checkpoints."""

import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torchvision import models


def init_weights(module):
    classname = module.__class__.__name__
    if "Conv2d" in classname or "ConvTranspose2d" in classname:
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif "BatchNorm" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif "Linear" in classname:
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


VGG_BUILDERS = {
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "vgg11bn": models.vgg11_bn,
    "vgg13bn": models.vgg13_bn,
    "vgg16bn": models.vgg16_bn,
    "vgg19bn": models.vgg19_bn,
}

RESNET_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50": models.resnext50_32x4d,
    "resnext101": models.resnext101_32x8d,
}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super().__init__()
        model_vgg = VGG_BUILDERS[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for index in range(6):
            self.classifier.add_module(
                f"classifier{index}", model_vgg.classifier[index]
            )
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        return self.classifier(outputs)


class ResBase(nn.Module):
    def __init__(self, res_name):
        super().__init__()
        model_resnet = RESNET_BUILDERS[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.maxpool(outputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        return outputs.view(outputs.size(0), -1)


class FeatureBottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, bottleneck_type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = bottleneck_type

    def forward(self, inputs):
        outputs = self.bottleneck(inputs)
        if self.type == "bn":
            outputs = self.bn(outputs)
        return outputs


class FeatureClassifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, classifier_type="linear"):
        super().__init__()
        self.type = classifier_type
        if classifier_type == "wn":
            self.fc = weight_norm(
                nn.Linear(bottleneck_dim, class_num), name="weight"
            )
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, inputs):
        return self.fc(inputs)


def build_models(config, device):
    backbone_name = config["model"]["backbone"]
    if backbone_name.startswith("res"):
        net_f = ResBase(backbone_name)
    elif backbone_name.startswith("vgg"):
        net_f = VGGBase(backbone_name)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    net_b = FeatureBottleneck(
        feature_dim=net_f.in_features,
        bottleneck_dim=config["model"]["bottleneck_dim"],
        bottleneck_type=config["model"]["bottleneck_type"],
    )
    net_c = FeatureClassifier(
        class_num=config["model"]["class_num"],
        bottleneck_dim=config["model"]["bottleneck_dim"],
        classifier_type=config["model"]["classifier_type"],
    )
    return net_f.to(device), net_b.to(device), net_c.to(device)


def load_source_models(config, device):
    import os.path as osp
    import torch

    net_f, net_b, net_c = build_models(config, device)
    checkpoint_dir = config["source_checkpoint"]["resolved_dir"]
    paths = {
        "netF": osp.join(checkpoint_dir, "source_F.pt"),
        "netB": osp.join(checkpoint_dir, "source_B.pt"),
        "netC": osp.join(checkpoint_dir, "source_C.pt"),
    }
    net_f.load_state_dict(torch.load(paths["netF"], map_location=device))
    net_b.load_state_dict(torch.load(paths["netB"], map_location=device))
    net_c.load_state_dict(torch.load(paths["netC"], map_location=device))
    return (net_f, net_b, net_c), paths
