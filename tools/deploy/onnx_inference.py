# coding=utf-8  
# @Time   : 2020/12/3 17:31
# @Auto   : zzf-jeff


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import onnxruntime
import numpy as np


class ONNXModel():
    def __init__(self, onnx_path):
        """

        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)

        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def to_numpy(self, tensor):
        if not isinstance(tensor,np.ndarray):
            tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def run(self, image):

        input_feed = self.get_input_feed(self.input_name, self.to_numpy(image))

        output = self.onnx_session.run(self.output_name, input_feed=input_feed)

        return output[0]
