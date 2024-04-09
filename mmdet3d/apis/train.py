import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook,
                         GradientCumulativeFp16OptimizerHook, OptimizerHook,
                         build_optimizer, build_runner)

from mmdet3d.runner import CustomEpochBasedRunner
from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet3d.core.evaluation.eval_hooks import CustomDistEvalHook
from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet3d.datasets import build_dataloader


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()
    # cfg.lr_config.target_ratio = tuple(cfg.lr_config.target_ratio)
    # cfg.momentum_config.target_ratio = tuple(cfg.momentum_config.target_ratio)
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            1,
            dist=distributed,
            seed=cfg.seed,
            nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
            shuffler_sampler=cfg.data.get("shuffler_sampler", None),
        )
        for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = model.cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )

    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # print('====================')
    # print(list2tuple(cfg.lr_config))
    # print('====================')
    # register hooks
    runner.register_training_hooks(
        list2tuple(cfg.lr_config),
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        list2tuple(cfg.get("momentum_config", None)),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=cfg.data.get("nonshuffler_sampler", None),
            shuffler_sampler=cfg.data.get("shuffler_sampler", None)
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        if cfg.data.get("nonshuffler_sampler", None):
            eval_hook = CustomDistEvalHook
        else:
            eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, [("train", 1)])


def list2tuple(cfg):
    # print(cfg)
    if cfg is not None:
        for i in cfg:
            if isinstance(cfg[i], list) and len(cfg[i]) == 2:
                cfg[i] = tuple(cfg[i])
    return cfg