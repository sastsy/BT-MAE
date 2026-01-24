# BT-MAE (Barlow-Twins Masked Autoencoder)

This repository includes a PyTorch implementation of BT-MAE, based on the repositories of the original [MAE](https://github.com/facebookresearch/mae) implementation and [U-MAE](https://github.com/zhangq327/U-MAE) implementation, which introduced a linear classifier to monitor online linear accuracy. 

BT-MAE is an extension of [MAE (He et al., 2022)](https://arxiv.org/pdf/2111.06377.pdf) by further encouraging the feature decorrelation of MAE.

## Instructions
We follow all the default training and evaluation configurations of MAE. Please see their instructions [README_mae.md](README_mae.md) for details.

**Main differences.** In BT-MAE, we introduce a ``bt_loss``  (implemented in ``models_mae.py``) as a decorrelating regularization to the MAE loss. It also introduces additional hyper-parameters ``bt_weight`` and ``bt_lambda`` in ``pretrain.sh``, which represent the coefficient of the Barlow-Twins regularization in the BT-MAE loss and ``lambda`` coefficient from original Barlow-Twins algorithm. 

## Training BT-MAE

To launch pretraining, the ``pretrain.sh`` script could be used:

```
EXP_NAME="bt_mae_pretrain"

torchrun --nproc_per_node=8 main_pretrain.py \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --use_hf_dataset \
    --lamb 0.00 \
    --reg none \
    --output_dir path/to/your/output/dir/$EXP_NAME \
    --log_dir path/to/your/logdir/$EXP_NAME \
    --bt_variant per_batch \
    --bt_weight 0.005 \
    --bt_lambda 0.0005
```

The ``bt_variant`` can be chosen from the list below. Note that each variant requires tuning the ``bt_weight`` and ``bt_lambda`` hyper-parameters.

To launch linear probing after pretraining, the ``linprobe.sh`` script could be used:

```
EXP_NAME="bt_mae_linear_probe"

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 main_linprobe.py \
    --accum_iter 4 \
    --batch_size 256 \
    --model vit_base_patch16 --cls_token\
    --finetune /path/to/your/checkpoint/checkpoint-199.pth \
    --epochs 90 \
    --blr 0.1  \
    --weight_decay 0.0 \
    --log_dir /path/to/your/logdir \
    --dist_eval --data_path data \
    --use_hf_dataset \
    --nb_classes 100 \
    --output_dir /path/to/your/output/dir/$EXP_NAME \

```

**Barlow-Twins variants:**
There are several variants of applying Barlow-Twins for feature decorrelation:
1. ``per_image``

Applies Barlow-Twins *independently per image*.
For each image, the cross-correlation matrix is computed across its patch tokens, encouraging decorrelation between tokens within the same image. Loss is averaged over the batch.

2. ``per_batch``

Applies Barlow-Twins *across the full batch of patch tokens*.
All patch tokens from all images are pooled (optionally across GPUs) and used to compute a single cross-correlation matrix, encouraging global feature decorrelation.

3. ``cls``

Applies Barlow-Twins on *CLS (or pooled) features only*.
The loss is computed across the batch using the global image representation, promoting decorrelated dimensions in the final image-level embedding.

4. ``cls_cross``

Cross-view Barlow-Twins on *CLS (or pooled) features from two views*.
Uses two different masked views of the same images and enforces invariance and decorrelation between their global representations.

5. ``per_image_cross``

Cross-view Barlow-Twins on patch tokens.
Patch tokens from two orthogonal masked views are matched, encouraging consistency and decorrelation between corresponding token representations across views.