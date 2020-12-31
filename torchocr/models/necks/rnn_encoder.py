# coding=utf-8  
# @Time   : 2020/12/18 17:25
# @Auto   : zzf-jeff

from torch import nn
import torch
from ..builder import NECKS


class Reshape(nn.Module):
    def __init__(self, **kwargs):
        super(Reshape, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # [w,b,c]
        return x

# 是否加入dropout
class BiLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channel, num_lstm, **kwargs):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(in_channels, hidden_channel, bidirectional=True, batch_first=True, num_layers=num_lstm)

    def forward(self, x):
        recurrent, (hn, cn) = self.rnn(x)
        return recurrent


@NECKS.register_module()
class EncodeWithLSTM(nn.Module):
    def __init__(self, num_lstm, in_channels, hidden_channel, **kwargs):
        super(EncodeWithLSTM, self).__init__()
        self.reshape = Reshape()
        self.num_lstm = num_lstm
        # self.out_channels = hidden_channel * 2
        self.lstm = BiLSTM(in_channels, hidden_channel, num_lstm)

    def forward(self, x):
        x = self.reshape(x)
        x = self.lstm(x)
        return x
