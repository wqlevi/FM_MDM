CUDA_VISIBLE_DEVICES=0,1,2,3 python runner_l.py \
    --n-gpu-per-node 4 --beta-max 0.3 --corrupt jpeg-5 \
    --num-itr 100 --dataset-dir "/kyb/agks/wangqi/CelebaHQ/celeba_hq_256/train/subclass_train/" \
    --log-writer 'tensorboard' --microbatch 2 --batch-size 60 --interval 1000 
