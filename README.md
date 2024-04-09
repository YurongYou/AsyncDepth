# Better Monocular 3D Detectors with LiDAR from the Past

[Paper](https://arxiv.org/pdf/2404.05139.pdf)

## Abstract
Accurate 3D object detection is crucial to autonomous driving. Though LiDAR-based detectors have achieved impressive performance, the high cost of LiDAR sensors precludes their widespread adoption in affordable vehicles. Camera-based detectors are cheaper alternatives but often suffer inferior performance compared to their LiDARbased counterparts due to inherent depth ambiguities in images. In this work, we seek to improve monocular 3D detectors by leveraging unlabeled historical LiDAR data. Speciﬁcally, at inference time, we assume that the camera-based detectors have access to multiple unlabeled LiDAR scans from past traversals at locations of interest (potentially from other high-end vehicles equipped with LiDAR sensors). Under this setup, we proposed a novel, simple, and end-to-end trainable framework, termed AsyncDepth, to effectively extract relevant features from asynchronous LiDAR traversals of the same location for monocular 3D detectors. We show consistent and signiﬁcant performance gain (up to 9 AP) across multiple state-of-the-art models and datasets with a negligible additional latency of 9.66 ms and a small storage cost.


## Environment
The codebase is built upon [BEVFusion](https://github.com/mit-han-lab/bevfusion).
Following the original codebase,
the code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- numba = 0.48.0
- numpy = 1.20.3
- [torchscatter](https://github.com/rusty1s/pytorch_scatter)
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)
- [ithaca365-dev-kit](https://github.com/cdiazruiz/ithaca365-devkit)
- yapf == 0.40.1 (see [here](https://github.com/open-mmlab/mmdetection/issues/10962))
- setuptools == 59.5.0 (see [here](https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version))

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

Additionally, install MinkowskiEngine
```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git \
    && cd MinkowskiEngine \
    && git checkout c854f0c \
    && python setup.py install
```

Alternatively, you can use the provided [Dockerfile](Dockerfile) to build the environment.

## Data Pre-processing

### Lyft Dataset
* Download the train set from [here](https://woven.toyota/en/perception-dataset).
* Untar the data into folder `LYFT_ROOT` and adjust the folder scructure into
    ```
    LYFT_ROOT
    ├── v1.01-train
        ├── images -> train_images
        ├── lidar -> train_lidar
        ├── maps -> train_maps
        ├── v1.01-train -> train_data
    ```
* Fix the LiDAR data issue by running
    ```bash
    python tools/data_converter/lyft_data_fixer.py --root-folder LYFT_ROOT
    ```
* Split the data into training and validation sets by running
    ```bash
    python lyft_data_split.py --root-folder LYFT_ROOT --prefix beta_v0_dist_20_cutoff_1000_ \
        --cutoff 1000 --max_distance 20 --upper_part_train --exclude_beta_plus_plus
    ```
* Run the data converter to generate the info files
    ```bash
    python tools/create_data.py --dataset lyft --version v1.01 --root-path LYFT_ROOT \
    --sample_info_prefix beta_v0_dist_20_cutoff_1000_ --extra-tag beta_v0_dist_20_cutoff_1000
    python tools/create_data.py --dataset lyft --version v1.01 --root-path LYFT_ROOT \
     --extra-tag beta_v0_dist_20_cutoff_1000 --gen-2d
    ```

### Itha365 Dataset
* Download the dataset from [here](https://ithaca365.mae.cornell.edu/).
* Run the following script to convert the dataset to the required format:
    ```bash
    python tools/create_data.py --root-path ITHACA_ROOT --dataset ithaca365 --extra-tag correct_history_v2_full
    python tools/create_data.py --root-path ITHACA_ROOT --dataset ithaca365 --extra-tag correct_history_v2_full --gen-2d
    ```

## Training Scripts
Download the pretrained models
```bash
bash tools/scripts/download_pretrained.sh
```
Run the following commands to train models with 4 GPUs.
### Lyft Dataset
* FCOS3D
    * w/ Async Depth

        1st stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/lyft_betav0_20_fcos3d/depth_hindsight_v2/max_gen_mean_op_bn_grad_pretrain_cp.yaml \
        --run-dir logs/lyft_betav0_20_v2_fcos3d+depth_cond+max_gen+mean_op+bn_grad+pretrain+cp \
        data.samples_per_gpu 4 data.workers_per_gpu 4
        ```

        2nd stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/lyft_betav0_20_fcos3d/depth_hindsight_v2/max_gen_mean_op_bn_grad_pretrain_cp_finetune.yaml \
        --run-dir logs/lyft_betav0_20_v2_fcos3d+depth_cond+max_gen+mean_op+bn_grad+pretrain+cp+finetune \
        data.samples_per_gpu 4 data.workers_per_gpu 4 \
        load_from logs/lyft_betav0_20_v2_fcos3d+depth_cond+max_gen+mean_op+bn_grad+pretrain+cp/latest.pth
        ```
    * w/o Async Depth

        1st stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/lyft_betav0_20_fcos3d/default.yaml \
        --run-dir logs/lyft_betav0_20_v2_fcos3d \
        data.samples_per_gpu 4 data.workers_per_gpu 4
        ```

        2nd stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/lyft_betav0_20_fcos3d/fine_tune.yaml \
        --run-dir logs/lyft_betav0_20_v2_fcos3d+finetune \
        data.samples_per_gpu 4 data.workers_per_gpu 4 \
        load_from logs/lyft_betav0_20_v2_fcos3d/latest.pth
        ```

* Lift-Splat
    * w/ Async Depth
        ``` bash
        torchpack dist-run -np 4 python \
        tools/train.py configs/lyft_betav0_20_v2/det/centerhead_5c/lssfpn/camera/384x800_50m/swint/depth_hindsight_resnet18_fpn_depth_filler_-1/50m_depth_sup_bn_grad_pretrain.yaml \
        --run-dir logs/lyft_new_split/lyft_4gpu_large_camera_only_beta_v0_dist_20_hindsight_depth_resnet18_fpn_bn_grad_pretrain_lr_warmup \
        model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        data.samples_per_gpu 2 \
        max_epochs 20 \
        evaluation.interval 2 \
        checkpoint_config.interval 2 \
        checkpoint_config.max_keep_ckpts 5 \
        optimizer.lr 1.0e-4
        ```
    * w/o Async Depth
        ```bash
        torchpack dist-run -np 4 python \
        tools/train.py configs/lyft_betav0_20_v2/det/centerhead_5c/lssfpn/camera/384x800_50m/swint/depth_sup_lr_linear_rampup.yaml \
        --run-dir logs/lyft_new_split/lyft_4gpu_large_camera_only_beta_v0_dist_20_depth_sup_lr_rampup \
        model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        data.samples_per_gpu 2 \
        max_epochs 20 \
        evaluation.interval 2 \
        checkpoint_config.interval 2 \
        checkpoint_config.max_keep_ckpts 5 \
        optimizer.lr 1.0e-4
        ```

### Ithaca365 Dataset
* FCOS3D
    * w/ Async Depth

        1st stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/ithaca365_fcos3d/depth_hindsight_v2/max_gen_mean_op_pretrained.yaml \
        --run-dir logs/ithaca365/fcos3d+depth_cond+max_gen+mean_op+pretrained
        ```

        2nd stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/ithaca365_fcos3d/depth_hindsight_v2/max_gen_mean_op_pretrained_finetune.yaml \
        --run-dir logs/ithaca365/fcos3d+depth_cond+max_gen+mean_op+pretrained+finetune \
        load_from logs/ithaca365/fcos3d+depth_cond+max_gen+mean_op+pretrained/latest.pth
        ```
    * w/ Async Depth

        1st stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/ithaca365_fcos3d/default.yaml \
        --run-dir logs/ithaca365/fcos3d
        ```

        2nd stage
        ```bash
        torchpack dist-run -np 4 python tools/train.py \
        configs/ithaca365_fcos3d/finetune.yaml \
        --run-dir logs/ithaca365/fcos3d+finetune \
        load_from logs/ithaca365/fcos3d/latest.pth
        ```
* Lift-Splat
    * w/ Async Depth
        ``` bash
        torchpack dist-run -np 4 python \
        tools/train.py configs/ithaca365/det/centerhead/lssfpn/camera/256x896/swint/depth_hindsight_resnet18_fpn_depth_filler_-1/50m_depth_sup_bn_grad_pretrain.yaml \
        --run-dir logs/ithaca365_v2/ithaca365_camera_hindsight_depth_resnet18_fpn_bn_grad_pretrain_lr_rampup \
        model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        data.samples_per_gpu 2 \
        evaluation.interval 2 \
        checkpoint_config.interval 2 \
        checkpoint_config.max_keep_ckpts 5 \
        optimizer.lr 1.0e-4 \
        max_epochs 20
        ```
    * w/o Async Depth
        ```bash
        torchpack dist-run -np 4 python \
        tools/train.py configs/ithaca365/det/centerhead/lssfpn/camera/256x896/swint/50m_depth_sup_lr_linear_rampup.yaml \
        --run-dir logs/ithaca365_v2/ithaca365_camera_depth_sup_lr_rampup \
        model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        data.samples_per_gpu 2 \
        evaluation.interval 2 \
        checkpoint_config.interval 2 \
        checkpoint_config.max_keep_ckpts 5 \
        optimizer.lr 1.0e-4 \
        max_epochs 20
        ```

    * w/ Sync Depth (Oracle)
        ```bash
        torchpack dist-run -np 4 python \
        tools/train.py configs/ithaca365/det/centerhead/lssfpn/camera/256x896/swint/depth_hindsight_resnet18_fpn_depth_filler_-1/50m_depth_sup_bn_grad_pretrain_gt_depth_conditioning.yaml \
        --run-dir logs/ithaca365_v2/ithaca365_camera_hindsight_depth_resnet18_fpn_bn_grad_pretrain_gt_depth_conditioning_lr_rampup \
        model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        data.samples_per_gpu 2 \
        evaluation.interval 2 \
        checkpoint_config.interval 2 \
        checkpoint_config.max_keep_ckpts 5 \
        optimizer.lr 1.0e-4 \
        max_epochs 20
        ```

## Evaluation
Use the corresponding config files and checkpoints to evaluate the models as follows:
```bash
torchpack dist-run -np 4 python tools/test.py <config_path> \
<ckpt_path> --eval bbox --eval-options eval_by_distance=true close_only=true
```
## Checkpoints

| Dataset    | Model      | Async Depth?        | ckpt | config |
|------------|------------|---------------------|------|------|
| Lyft       | FCOS3D     | ✅                   |[link](https://drive.google.com/file/d/1LY6my6VKozhAmVifJLFA3SYvdE64t5NL/view?usp=drive_link)|[config](configs/lyft_betav0_20_fcos3d/depth_hindsight_v2/max_gen_mean_op_bn_grad_pretrain_cp_finetune.yaml)|
|            | Lift-Splat | ✅                   |[link](https://drive.google.com/file/d/1NfNuuJDTnSnqq4JrvTlPHndwk1Yc3Kwd/view?usp=drive_link)|[config](configs/lyft_betav0_20_v2/det/centerhead_5c/lssfpn/camera/384x800_50m/swint/depth_hindsight_resnet18_fpn_depth_filler_-1/50m_depth_sup_bn_grad_pretrain.yaml)|
|            | Lift-Splat | ❌                   |[link](https://drive.google.com/file/d/15AFwKBSfHuTIIS02l7HE7jicI18JAJSu/view?usp=drive_link)|[config](configs/lyft_betav0_20_v2/det/centerhead_5c/lssfpn/camera/384x800_50m/swint/depth_sup_lr_linear_rampup.yaml)|
| Ithaca-365 | FCOS3D     | ✅                   |[link](https://drive.google.com/file/d/1Lco2EGbbW_B_Y0YfPUtCHV1Zg0AIgid8/view?usp=drive_link)|[config](configs/ithaca365_fcos3d/depth_hindsight_v2/max_gen_mean_op_pretrained_finetune.yaml)|
|            | Lift-Splat | ✅                   |[link](https://drive.google.com/file/d/1D8QDxxfQzMBoUb4QqlicWeQtoxnaf7f_/view?usp=drive_link)|[config](configs/ithaca365/det/centerhead/lssfpn/camera/256x896/swint/depth_hindsight_resnet18_fpn_depth_filler_-1/50m_depth_sup_bn_grad_pretrain.yaml)|
|            | Lift-Splat | ❌                   |[link](https://drive.google.com/file/d/1Aum4WF9KeZ-5or6-SCy26dpvihFu9mRk/view?usp=drive_link)|[config](configs/ithaca365/det/centerhead/lssfpn/camera/256x896/swint/50m_depth_sup_lr_linear_rampup.yaml)|
|            | Lift-Splat | Sync-Depth (Oracle) |[link](https://drive.google.com/file/d/1stvDLd53E2MXIx57ux7eGWfYPGjTK_qR/view?usp=drive_link)|[config](configs/ithaca365/det/centerhead/lssfpn/camera/256x896/swint/depth_hindsight_resnet18_fpn_depth_filler_-1/50m_depth_sup_bn_grad_pretrain_gt_depth_conditioning.yaml)|

## Contact
Please open an issue if you have any questions about using this repo.

## Acknowledgement
This work is based on [BEVFusion](https://github.com/mit-han-lab/bevfusion) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). We also use [MinkowskiEngine
](https://github.com/NVIDIA/MinkowskiEngine).
We thank them for open-sourcing excellent libraries for 3D understanding tasks.

## Citation
```
@inproceedings{you2024better,
  title = {Better Monocular 3D Detectors with LiDAR from the Past},
  author = {You, Yurong and Phoo, Cheng Perng and Diaz-Ruiz, Carlos Andres and Luo, Katie Z and Chao, Wei-Lun and Campbell, Mark  and Hariharan, Bharath and Weinberger, Kilian Q},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2024},
  month = jun,
}
```