# coding=utf-8  
# @Time   : 2020/12/2 18:55
# @Auto   : zzf-jeff
from .base_converter import BaseConverter
from .builder import CONVERTER
import torch


def get_keys(key_path):
    with open(key_path, 'r', encoding='utf-8') as fid:
        lines = fid.readlines()[0]
        lines = lines.strip('\n')
        return lines


@CONVERTER.register_module()
class CTCConverter(BaseConverter):
    def __init__(self, alphabet_path):
        self.alphabet = get_keys(alphabet_path)
        # for `-1` index
        self.alphabet = self.alphabet + '-'
        print(self.alphabet)
        super(CTCConverter, self).__init__(self.alphabet)

    def encode(self, text):
        """Support batch or single str.

        :param text: text (str or list of str): texts to convert.
        :return:  torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
                  torch.IntTensor [n]: length of each text.
        """
        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False
        for item in text:
            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                idx = self.dict[char]
                result.append(idx)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        :param t: torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
        :param length: torch.IntTensor [n]: length of each text.
        :param raw:
        :return: text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
