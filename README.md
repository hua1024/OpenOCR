# OpenOCR
## 简介
- 通过自己搭建OCR的框架，总结学习到的OCR算法、pytorch的操作、python操作、部署操作
- 同步添加部分通用目标检测模型供OCR场景使用
- 性能正在逐步优化




## 更新日志

- 添加amp和ema并测试通过 2021.04.17
- TensorRt加速CRNN和DBNet测试通过 2021.03.22
- DDP单机多卡测试通过 2021.01.18
- 添加主干网络(ResNetVd、MobileNetV3) 2020.12.26
- CRNN测试通过 2020.12.15
- DBNet测试通过 2020.12.10
- 框架训练测试推理测试通过 2020.11.30




## 目前已支持
- [x] DBNet 
- [x] CRNN

## 复现记录
The ocr detect icdar2015 results should be as follows:

|        Model       	| precision(ours) 	| recall(ours) 	| F-measure(ours) 	| precision (paper) 	| recall (paper) 	| F-measure (paper) 	|
|:------------------:	|:---------:	|:------:	|:---------:	|:-----------------:	|:--------------:	|:-----------------:	|
| DBNet-r50 	|    88.9   	|  77.6  	|    82.9   	|        88.3       	|      77.9      	|        82.8       	|

The ocr recognition origin data results should be as follows:

|        Model       	| precision(ours) 	| norm_edit_dis(ours) 	| F-precision(origin) 	| norm_edit_dis (paper) 	|
|:------------------:	|:---------:	|:------:	|:---------:	|:-----------------:	|
| CRNN-r50-2lstm 	|    88.9   	|  77.6  	|    82.9   	|        88.3       	|


## TRT加速效果


### reference
    1.https://github.com/open-mmlab/mmdetection
    2.https://github.com/PaddlePaddle/PaddleOCR
> If this repository helps you，please star it. Thanks.

