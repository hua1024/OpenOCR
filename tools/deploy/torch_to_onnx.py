# coding=utf-8  
# @Time   : 2020/10/27 17:09
# @Auto   : zzf-jeff

import sys
import os

sys.path.append('./')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import onnx
import torch
import argparse
import cv2
from torchocr.utils.config_util import Config
from torchocr.models import build_model
from torchocr.utils.checkpoints import load_checkpoint
from torchocr.postprocess import build_postprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Torch to onnx')
    parser.add_argument('--config', type=str, help='train config file path', default='config/resnet50.yaml')
    parser.add_argument('--weights', type=str, help='the model weights')
    parser.add_argument('--onnx_output', type=str, help='onnx save path')
    parser.add_argument('--mode', type=str, default='rec', help='rec/det')
    parser.add_argument('--is_dynamic', action='store_true')
    parser.add_argument('--input_shape', type=str, default='1, 3, 32, 200', help='onnx save path')

    args = parser.parse_args()
    return args


def torch2onnx(model, dummy_input, onnx_model_name, input_names, output_names, opset_version=12,
               do_constant_folding=False, verbose=False, is_dynamic=False, dynamic_axes=None):
    """convert PyTorch model to Onnx
    主要以官方API为准
    Args:
        model (torch.nn.Module): PyTorch model.
        dummy_input (torch.Tensor, tuple or list): dummy input.
        onnx_model_name (string or io object): saved Onnx model name.
        opset_version (int, default is 9): Onnx opset version.
        do_constant_folding (bool, default False): 是否执行常量折叠优化,If True, the
            constant-folding optimization is applied to the model during
            export. Constant-folding optimization will replace some of the ops
            that have all constant inputs, with pre-computed constant nodes.
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        is_dynamic : 批处理变量,指定则使用批处理或者固定批处理大小，默认不使用
    """
    if is_dynamic:
        dynamic_axes = dynamic_axes
    else:
        dynamic_axes = None

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        dynamic_axes=dynamic_axes)
    return onnx_model_name


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    # set pretrained model None
    cfg.model.pretrained = None

    # build postprocess
    postprocess = build_postprocess(cfg.postprocess)
    # for rec cal head number
    if hasattr(postprocess, 'character'):
        char_num = len(getattr(postprocess, 'character'))
        cfg.model.head.n_class = char_num

    # use config build model
    model = build_model(cfg.model)

    # set weights to model and set model to device/eval()
    model_path = args.weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    load_checkpoint(model, model_path, map_location=device)
    model = model.to(device)
    model.eval()

    onnx_output = args.onnx_output
    input_shape = args.input_shape
    input_shape = eval(input_shape)
    mode = args.mode


    # transform torch model to onnx model
    input_names = ['input']
    output_names = ['output']

    # input shape
    input_data = torch.randn(input_shape).to(device)

    if args.is_dynamic:
        if mode == 'rec':
            # #rec
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        elif mode == 'det':
            ## det
            dynamic_axes = {"input": {0: "batch_size", 2: 'height', 3: 'width'},
                            "output": {0: "batch_size", 2: 'height', 3: 'width'}}

    else:
        dynamic_axes = None

    onnx_model_name = torch2onnx(
        model=model,
        dummy_input=input_data,
        onnx_model_name=onnx_output,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        is_dynamic=args.is_dynamic,
        dynamic_axes=dynamic_axes
    )

    onnx_model = onnx.load(onnx_model_name)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    print("Model was successfully converted to ONNX format.")
    print("It was saved to", onnx_model_name)


if __name__ == '__main__':
    main()
