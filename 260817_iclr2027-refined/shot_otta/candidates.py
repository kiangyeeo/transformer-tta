"""Frozen parameter candidate definitions shared by OTTA methods."""

MODULE_CANDIDATE_ORDER = (
    "netB.bottleneck.weight",
    "netB.bottleneck.bias",
)
MODULE_CANDIDATE_NAMES = set(MODULE_CANDIDATE_ORDER)

CONV_CANDIDATE_ORDER = (
    "netF.layer4.0.conv1.weight",
    "netF.layer4.0.conv2.weight",
    "netF.layer4.0.conv3.weight",
    "netF.layer4.1.conv1.weight",
    "netF.layer4.1.conv2.weight",
    "netF.layer4.1.conv3.weight",
    "netF.layer4.2.conv1.weight",
    "netF.layer4.2.conv2.weight",
    "netF.layer4.2.conv3.weight",
)
CONV_CANDIDATE_NAMES = set(CONV_CANDIDATE_ORDER)
CONV_CANDIDATE_SHAPES = {
    "netF.layer4.0.conv1.weight": (512, 1024, 1, 1),
    "netF.layer4.0.conv2.weight": (512, 512, 3, 3),
    "netF.layer4.0.conv3.weight": (2048, 512, 1, 1),
    "netF.layer4.1.conv1.weight": (512, 2048, 1, 1),
    "netF.layer4.1.conv2.weight": (512, 512, 3, 3),
    "netF.layer4.1.conv3.weight": (2048, 512, 1, 1),
    "netF.layer4.2.conv1.weight": (512, 2048, 1, 1),
    "netF.layer4.2.conv2.weight": (512, 512, 3, 3),
    "netF.layer4.2.conv3.weight": (2048, 512, 1, 1),
}
