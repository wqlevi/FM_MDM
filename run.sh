CUDA_VISIBLE_DEVICES=0,1,2,3 python train_mp.py \
    --n-gpu-per-node 4 --beta-max 1.0 --corrupt 'inpaint-center' --name 'inpaint_256_madeira' \
    --num-itr 6001 --dataset-dir "/kyb/agks/wangqi/CelebaHQ/celeba_hq_256/" \
    --log-writer 'wandb' \
    --wandb-api-key e8b8669b01462d8329d90ab655789f2e0e203ca8 --wandb-user 'wqlevi' 
