# coding=utf-8  
# @Time   : 2020/12/2 9:40
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import HEADS


# class BLSTM(nn.Module):
#     def __init__(self, in_channel, hidden_channel, output_channel):
#         super(BLSTM, self).__init__()
#         self.rnn = nn.LSTM(in_channel, hidden_channel, bidirectional=True)
#         self.embedding = nn.Linear(hidden_channel * 2, output_channel)
#
#     def forward(self, x):
#         recurrent, (hn, cn) = self.rnn(x)
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)
#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#         return output


@HEADS.register_module()
class CTCHead(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super(CTCHead, self).__init__()
        self.embedding = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        T, b, h =  x.size()
        t_rec =  x.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

#
# @HEADS.register_module()
# class CRNNHead(nn.Module):
#     def __init__(self, use_conv=False, use_attention=False, use_lstm=True,
#                  lstm_num=2, in_channel=512, hidden_channel=256, classes=1000):
#         """Crnn head
#
#         :param use_conv: cnn-->conv2 classes, if not use lstm, have conv or linear to classes
#         :param use_attention: cnn-->attention
#         :param use_lstm: cnn-->lstm
#         :param lstm_num: number of lstm
#         :param in_channel: input channel,512 or
#         :param hidden_channel: in_channel//2
#         :param classes: number label + blank
#         """
#         super(CRNNHead, self).__init__()
#         self.use_lstm = use_lstm
#         self.lstm_num = lstm_num
#         self.use_conv = use_conv
#         self.use_attention = use_attention
#         self.in_channel = in_channel
#         self.hidden_channel = hidden_channel
#         self.classes = classes
#         self.init_lstm()
#         self.init_conv()
#         self.init_atten()
#
#     def init_atten(self):
#         if self.use_attention:
#             self.attention = None
#         else:
#             self.attention = None
#
#     def init_lstm(self):
#         if (self.use_lstm):
#             assert self.lstm_num > 0, Exception('lstm_num need to more than 0 if use_lstm = True')
#             for i in range(self.lstm_num):
#                 if i == 0:
#                     if (self.lstm_num == 1):
#                         setattr(self, 'lstm_{}'.format(i + 1),
#                                 BLSTM(self.in_channel, self.hidden_channel, self.classes))
#                     else:
#                         setattr(self, 'lstm_{}'.format(i + 1),
#                                 BLSTM(self.in_channel, self.hidden_channel, self.hidden_channel))
#                 elif (i == self.lstm_num - 1):
#                     setattr(self, 'lstm_{}'.format(i + 1),
#                             BLSTM(self.hidden_channel, self.hidden_channel, self.classes))
#                 else:
#                     setattr(self, 'lstm_{}'.format(i + 1),
#                             BLSTM(self.hidden_channel, self.hidden_channel, self.hidden_channel))
#
#     def init_conv(self):
#         if (self.use_conv):
#             self.out = nn.Conv2d(self.in_channel, self.classes, kernel_size=1, padding=0)
#         else:
#             self.out = nn.Linear(self.in_channel, self.classes)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         assert h == 1, "the height of conv must be 1"
#         if self.use_attention:
#             x = self.attention(x)
#         if (self.use_conv):
#             x = self.out(x)
#             x = x.squeeze(2)
#             x = x.permute(2, 0, 1)
#             return x
#         x = x.squeeze(2)
#         x = x.permute(2, 0, 1)  # [w, b, c]
#         if self.use_lstm:
#             for i in range(self.lstm_num):
#                 x = getattr(self, 'lstm_{}'.format(i + 1))(x)
#         else:
#             x = self.out(x)
#         return x
