# coding=utf-8  
# @Time   : 2020/12/28 15:06
# @Auto   : zzf-jeff
import os

def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)