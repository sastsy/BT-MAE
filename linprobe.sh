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
    --log_dir /path/to/your/log/dir/$EXP_NAME
