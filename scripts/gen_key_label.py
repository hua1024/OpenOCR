# coding=utf-8  
# @Time   : 2021/5/11 11:20
# @Auto   : zzf-jeff


train_list_file = './test_list.txt'
test_list_file = './train_list.txt'
keys_file = './key.txt'

fid_key = open(keys_file,'w+',encoding='utf-8')
keys = ''
with open(train_list_file,'r',encoding='utf-8') as fid_train:
    lines = fid_train.readlines()
    for line in lines:
        line = line.strip().split('\t')
        keys+=line[-1]

with open(test_list_file,'r',encoding='utf-8') as fid_test:
    lines = fid_test.readlines()
    for line in lines:
        line = line.strip().split('\t')
        keys+=line[-1]

key = ''.join(list(set(list(keys))))
fid_key.write(key)