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
