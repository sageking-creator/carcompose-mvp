import torch.nn as nn


def cast_to_fp16(model: nn.Module) -> nn.Module:
    """
    Cast model to FP16. Norm layers remain FP32 to avoid numerical instability.
    """

    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            module.float()
        else:
            module.half()
    return model

