CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mp.py \
    --n-gpu-per-node 4 --beta-max 1.0 --corrupt 'jpeg-5' --name 'i2sb_128' \
    --num-itr 17050 --dataset-dir "/kyb/agks/wangqi/CelebaHQ/celeba_hq_256/" \
    --log-writer 'wandb' --microbatch 8 --batch-size 64 --image-size 128 --interval 1000 \
    --wandb-api-key e8b8669b01462d8329d90ab655789f2e0e203ca8 --wandb-user 'wqlevi' --ckpt 'i2sb_128'
