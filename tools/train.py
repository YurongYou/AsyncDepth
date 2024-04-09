import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

import wandb
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import (convert_sync_batchnorm, get_root_logger,
                           recursive_eval)

def cfg_clean_up(cfg):
    for transform in cfg.data.train.pipeline:
        if transform.type == 'Resize':
            transform.img_scale = tuple(transform.img_scale)
    for transform in cfg.data.val.pipeline:
        if transform.type == 'MultiScaleFlipAug':
            for _transform in transform.transforms:
                if _transform.type == 'Resize':
                    _transform.img_scale = tuple(_transform.img_scale)

def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--wandb_project", default=None, help="wandb_project")
    parser.add_argument("--wandb_id", default=None, help="wandb_id")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark # pyright: ignore[reportGeneralTypeIssues]
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    cfg_clean_up(cfg)

    if dist.local_rank() == 0 and args.wandb_project is not None:
        # hava to create this before calling tensorboard
        if args.wandb_id is not None and cfg.resume_from is not None:
            wandb.init(project=args.wandb_project,
                    name=os.path.basename(cfg.run_dir),
                    id=args.wandb_id,
                    sync_tensorboard=True,
                       resume="allow"
                    )
        else:
            wandb.init(project=args.wandb_project,
                    name=os.path.basename(cfg.run_dir),
                    sync_tensorboard=True,
                    )
        wandb.config.update(cfg, allow_val_change=True)

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True # pyright: ignore[reportGeneralTypeIssues]
            torch.backends.cudnn.benchmark = False  # pyright: ignore[reportGeneralTypeIssues]
    # print(cfg.data.train)
    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    if dist.local_rank() == 0 and args.wandb_project is not None:
        wandb.config.update({"total_step": len(datasets[0]) * cfg.max_epochs // cfg.data.samples_per_gpu // dist.size()})
        # wandb.watch(model)
    distributed = torch.distributed.is_initialized()
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
