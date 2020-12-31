# coding=utf-8  
# @Time   : 2020/12/29 14:10
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
import math
import time

# 设定SEED，让之后随机数生成一致
SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)
# torch.backends.cudnn.benchmark = True 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销。
torch.backends.cudnn.deterministic=True
spacy_de=spacy.load('de')
spacy_en=spacy.load('en')

def tokenize_de(text):
    # 使用[::-1]将文本进行倒序排列
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC=Field(
    tokenize=tokenize_de,
    init_token='<sos>',
    eos_token='<eos>',
    lower=True
)
TRG=Field(
    tokenize=tokenize_en,
    init_token='<sos>',
    eos_token='<eos>',
    lower=True
)

train_data, valid_data, test_data=Multi30k.splits(exts=('.de','.en'),fields=(SRC,TRG))

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")
print(vars(train_data.examples[1]))
SRC.build_vocab(train_data,min_freq=2)
TRG.build_vocab(train_data,min_freq=2)
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")