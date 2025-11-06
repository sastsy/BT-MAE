EXP_NAME="8gpu_per_batch_mae_pretrain_SMI"

torchrun --nproc_per_node=8 /home/jovyan/shares/SR004.nfs2/aitsybina/reps/malvina-assessor-mfu/main_pretrain.py \
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
    --output_dir "/home/jovyan/shares/SR004.nfs2/aitsybina/reps/malvina-assessor-mfu/train_output/$EXP_NAME" \
    --log_dir /home/jovyan/shares/SR004.nfs2/aitsybina/reps/malvina-assessor-mfu/train_logs/${EXP_NAME} \
    --bt_variant per_batch \
    --bt_weight 0.005 \
    --bt_lambda 0.005
    # --distributed
