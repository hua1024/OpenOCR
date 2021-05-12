# coding=utf-8  
# @Time   : 2021/5/11 11:20
# @Auto   : zzf-jeff


train_list_file = '../test/train_rec_05.txt'
test_list_file = '../test/test_rec_05.txt'
keys_file = './key.txt'

fid_key = open(keys_file, 'a+', encoding='utf-8')
keys = ''


def read_txt(txt_path, split_type):
    '''
    读取txt文件的标注信息，格式为
    xxx/a/1.png,a
    xxx/a/2.png,a
    Args:
        txt_path: train/valid/test data txt or json
    Returns:
        imgs：list, all data info
    '''
    with open(txt_path, 'r', encoding='utf-8') as f:
        infos = list(map(lambda line: line.strip().split(split_type), f))
    return infos


infos = read_txt(train_list_file, split_type=' ')
for info in infos:
    txt_file, use_flag = info
    if int(use_flag) == 1:
        with open(txt_file, 'r', encoding='utf-8') as fid_train:
            lines = fid_train.readlines()
            for line in lines:
                line = line.strip().split('\t')
                keys += line[-1]

infos = read_txt(test_list_file, split_type=' ')
for info in infos:
    txt_file, use_flag = info
    if int(use_flag) == 1:
        with open(txt_file, 'r', encoding='utf-8') as fid_train:
            lines = fid_train.readlines()
            for line in lines:
                line = line.strip().split('\t')
                keys += line[-1]

key = ''.join(list(set(list(keys))))
for _key in key:
    fid_key.write(_key + '\n')
