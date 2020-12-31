# coding=utf-8  
# @Time   : 2020/12/28 11:08
# @Auto   : zzf-jeff

import os
import os.path as osp


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    # expanduser : 展开~成全路劲
    dir_name = osp.expanduser(dir_name)
    if '.' in dir_name:
        dir_name = os.path.dirname(dir_name)
    if os.path.exists(dir_name):
        return
    os.makedirs(dir_name, mode=mode, exist_ok=True)
