# coding=utf-8  
# @Time   : 2020/12/18 17:25
# @Auto   : zzf-jeff

from torch import nn
from ..builder import NECKS


@NECKS.register_module()
class Im2Seq(nn.Module):
    def __init__(self, **kwargs):
        super(Im2Seq, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(axis=2)  # [b,c,T]
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channel)s
        return x


class Reshape(nn.Module):
    def __init__(self, **kwargs):
        super(Reshape, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape  # [b,c,1,w]
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2)  # [b,c,w]
        x = x.permute(2, 0, 1)  # [w,b,c]
        return x


class BiLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channel, num_lstm, drop=0.3, **kwargs):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(
            in_channels,
            hidden_channel,
            bidirectional=True,
            num_layers=num_lstm,
            batch_first=True,  # if batch_first=True ,out is seq
            dropout=drop
        )

    def forward(self, x):
        # UserWarning: RNN module weights are not part of single contiguous chunk of memory
        self.rnn.flatten_parameters()
        recurrent, (hn, cn) = self.rnn(x)
        # recurrent --> [T,B,C]
        return recurrent


class BiGRU(nn.Module):
    def __init__(self, in_channels, hidden_channel, num_lstm, drop=0.3, **kwargs):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(
            in_channels,
            hidden_channel,
            bidirectional=True,
            num_layers=num_lstm,
            batch_first=True,  # if batch_first=True ,out is seq
            dropout=drop
        )

    def forward(self, x):
        # UserWarning: RNN module weights are not part of single contiguous chunk of memory
        self.rnn.flatten_parameters()
        recurrent, (hn, cn) = self.rnn(x)
        # recurrent --> [T,B,C]
        return recurrent


@NECKS.register_module()
class EncodeWithLSTM(nn.Module):
    def __init__(self, num_lstm, in_channels, hidden_channel, **kwargs):
        super(EncodeWithLSTM, self).__init__()
        self.reshape = Im2Seq()
        self.num_lstm = num_lstm
        self.lstm = BiLSTM(in_channels, hidden_channel, num_lstm)

    def forward(self, x):
        x = self.reshape(x)
        x = self.lstm(x)
        return x


@NECKS.register_module()
class EncodeWithGRU(nn.Module):
    def __init__(self, num_lstm, in_channels, hidden_channel, **kwargs):
        super(EncodeWithGRU, self).__init__()
        self.reshape = Im2Seq()
        self.num_lstm = num_lstm
        self.lstm = BiGRU(in_channels, hidden_channel, num_lstm)

    def forward(self, x):
        x = self.reshape(x)
        x = self.lstm(x)
        return x


@NECKS.register_module()
class EncodeWithFC(nn.Module):
    def __init__(self, in_channels, hidden_channel, **kwargs):
        super(EncodeWithFC, self).__init__()
        self.reshape = Im2Seq()
        self.fc = nn.Linear(in_channels, hidden_channel)

    def forward(self, x):
        x = self.reshape(x)
        output = self.fc(x)
        return output



