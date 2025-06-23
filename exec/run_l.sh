CUDA_VISIBLE_DEVICES=0,1,2,3 python runner_l.py \
    --n-gpu-per-node 4 --beta-max 0.3 --corrupt jpeg-5 --name 'i2sb_l' \
    --num-itr 550 --dataset-dir "/kyb/agks/wangqi/CelebaHQ/celeba_hq_256/" \
    --log-writer 'wandb' --microbatch 6 --batch-size 60 --image-size 256 --interval 1000 \
