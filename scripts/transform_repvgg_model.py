# coding=utf-8  
# @Time   : 2021/3/9 10:57
# @Auto   : zzf-jeff


'''
if use repvgg as backbone, must using script to transform model

'''

import torch
import numpy as np


def whole_model_convert(train_model: torch.nn.Module, deploy_model: torch.nn.Module):
    all_weights = {}

    for name, module in train_model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            all_weights[name + '.rbr_reparam.weight'] = kernel
            all_weights[name + '.rbr_reparam.bias'] = bias
            print('convert RepVGG block')
        else:
            for p_name, p_tensor in module.named_parameters():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    # all_weights[full_name] = p_tensor.detach().cpu().numpy()
                    all_weights[full_name] = p_tensor

            for p_name, p_tensor in module.named_buffers():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    # all_weights[full_name] = p_tensor.cpu().numpy()
                    all_weights[full_name] = p_tensor

    load_weight_dict(deploy_model, all_weights)

    return deploy_model


def repvgg_model_convert(model: torch.nn.Module, build_func, save_path=None):
    converted_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + '.rbr_reparam.weight'] = kernel
            converted_weights[name + '.rbr_reparam.bias'] = bias
        elif isinstance(module, torch.nn.Linear):
            converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
            converted_weights[name + '.bias'] = module.bias.detach().cpu().numpy()
    del model

    deploy_model = build_func(deploy=True)
    for name, param in deploy_model.named_parameters():
        print('deploy param: ', name, param.size(), np.mean(converted_weights[name]))
        param.data = torch.from_numpy(converted_weights[name]).float()

    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.

    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank = 0
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_weight_dict(model, state_dict, map_location=None, strict=False, logger=None):
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict)
    else:
        load_state_dict(model, state_dict, strict)


def main():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
    from torchocr.models import build_model
    from torchocr.utils.config_util import Config
    from torchocr.utils.checkpoints import load_checkpoint, save_checkpoint

    cfg_path = 'config/det/dbnet/61_hw_repb2_dbnet.py'
    model_path = 'work_dirs/61_hw_repb2_dbnet/best.pth'

    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # build model
    train_model = build_model(cfg.model)
    load_checkpoint(train_model, model_path, map_location=device)
    train_model = train_model.to(device)

    cfg.model.backbone.is_deploy = True
    deploy_model = build_model(cfg.model)
    deploy_model = deploy_model.to(device)

    deploy_weights = whole_model_convert(train_model, deploy_model)
    save_checkpoint(deploy_weights, filepath='db_repvgg.pth')


if __name__ == '__main__':
    main()
