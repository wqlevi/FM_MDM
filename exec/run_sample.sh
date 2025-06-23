CUDA_VISIBLE_DEVICES=0,1,2,3 python sample.py \
    --n-gpu-per-node 4 \
    --ckpt 'raven_inpaint_ddp' --image-size 256 --dataset-dir "/kyb/agks/wangqi/CelebaHQ/celeba_hq_256/" \
    --batch-size 8 --nfe 200 
