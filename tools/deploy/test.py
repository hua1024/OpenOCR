# coding=utf-8  
# @Time   : 2021/3/6 9:12
# @Auto   : zzf-jeff
import argparse
from typing import Tuple, List

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time



def test1():
    from tools.deploy.onnx_inference import ONNXModel
    from torchvision import transforms
    from PIL import Image

    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    img = Image.open('test/test2.png')
    img = img.convert('RGB')
    img = trans(img)
    img = img.unsqueeze(0)
    img_numpy = np.array(img, dtype=np.float32, order='C')
    # img_numpy = np.concatenate((img_numpy, img_numpy), axis=0)


    onnx = ONNXModel('d50-sim.onnx')

    out = onnx.run(img_numpy)
    print(out)
    # trt = TRTModel('dynamic_d50.engine')
    # output = trt.run(img_numpy)
    # output_data = torch.Tensor(output)
    # tuple_t = output_data.split(split_size=1, dim=0)
    #
    # for out in tuple_t:
    #     prob = F.softmax(out, dim=1)
    #     print(prob)
    #     value, predicted = torch.max(out.data, 1)
    #     pred_class = ['dog', 'cat'][predicted.item()]
    #     pred_score = prob[0][predicted.item()].item()
    #     print(pred_class, pred_score)


test1()
