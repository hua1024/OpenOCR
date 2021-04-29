# coding=utf-8  
# @Time   : 2021/3/8 16:13
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)

resnet_layer = nn.Sequential(*list(resnet18.children())[:-2])
print(resnet_layer)

# parm = {}
# for name, parameters in resnet18.named_parameters():
#     print(name, ':', parameters.size())
#     parm[name] = parameters.detach().numpy()