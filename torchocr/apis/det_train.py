# coding=utf-8  
# @Time   : 2020/12/4 9:49
# @Auto   : zzf-jeff

from ..utils.logger import get_logger
from ..datasets import build_det_dataloader, build_det_dataset
from ..runners.epoch_based_runner import EpochBasedRunner
from ..optimizers import build_optimizer
from ..lr_schedulers import build_lr_scheduler
from ..hooks.eval_hooks import EvalHook


def train_detector(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    logger = get_logger(cfg.log_level)

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_det_dataloader(ds, data=cfg.data) for ds in dataset
    ]

    optimizer = build_optimizer(cfg.optimizer, model)
    # lr_scheduler = build_lr_scheduler(lr)(optimizer)
    runner = EpochBasedRunner(model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    ## register eval hooks 需要放在日志前面，不然打印不出日志。
    if validate:
        cfg.data.val.train = False
        val_dataset = build_det_dataset(cfg.data.val)
        val_dataloader = build_det_dataloader(val_dataset, shuffle=False, data=cfg.data)
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(EvalHook(val_dataloader, **eval_cfg))

    runner.register_training_hooks(cfg.checkpoint_config, cfg.log_config)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
