# coding=utf-8  
# @Time   : 2020/12/4 10:31
# @Auto   : zzf-jeff


import os
import time
import torch
import random
import numpy as np


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device='', apex=False, batch_size=None):

    # todo : device的冗余处理可以去掉

    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'

    if device and not cpu_request:  # if device requested other than 'cpu'
        # os.environ['CUDA_VISIBLE_DEVICES'] 有很多情况下会失效，只有在主程序train.py会生效
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity
    cuda = False if cpu_request else torch.cuda.is_available()

    # concat device id
    str_ids = device.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    # print device param
    if cuda:
        c = 1024 ** 2  # bytes to MB
        device = device.replace(',', '')
        ng = len(device.replace(',', ''))
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, len(device)):
            if i == 1:
                s = ' ' * len(s)
            print("=> %sdevice%s _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, device[i], x[i].name, x[i].total_memory / c))
    else:
        print('=> Using CPU')
    print('')  # skip a line

    return torch.device('cuda:{}'.format(gpu_ids[0]) if cuda else 'cpu'), gpu_ids


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report == 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))
