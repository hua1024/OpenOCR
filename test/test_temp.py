import os

from tqdm import tqdm

label_file = 'print_ocr.txt'
res_txt = 'print_ocr_new.txt'

with open(label_file, 'r', encoding='utf-8') as f:
    for line in tqdm(f.readlines(), desc='load data from {}'.format(label_file)):
        for li in line:
            with open(res_txt, 'a+', encoding='utf-8') as fw:
                fw.write(li+'\n')