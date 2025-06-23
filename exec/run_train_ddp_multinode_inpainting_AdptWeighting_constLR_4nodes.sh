#!/bin/bash -l
##standard output and error:
#SBATCH -o ./log/i2sb.out.%j
#SBATCH -e ./log/i2sb.err.%j
# initial working dir:
#SBATCH -D ./
# Job name:
#SBATCH -J I2SB_micro32_W
# Node feature
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=8
### SBATCH --ntasks=4
## #SBATCH --mem=20G
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
# Number of nodes and MPI tasks per node:
# wall clock limit(Max. is 24hrs)
#SBATCH --time=24:00:00
#SBATCH --mail-user=qi.wang@tuebingen.mpg.de

module purge 
module load anaconda/3/2021.11
module load gcc/11
module load openmpi/4
#module load impi/2021.7
module load pytorch-distributed/gpu-cuda-11.6/2.0.0
module load pytorch-lightning/2.0.1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export LOGLEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=12316
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
srun hostname > hostfile.txt


srun python train_debug.py --num-proc-node 8 --n-gpu-per-node 4 --name 'raven_FM_MDM_AdptWeighting_ConstLR_micro32_corrected_large' --corrupt 'inpaint-center' --num-itr 200000 --batch-size 256 --dataset-dir '/ptmp/wangqi/celeba_hq_256/' --microbatch 2 --log-writer 'wandb' --wandb-api-key 'e8b8669b01462d8329d90ab655789f2e0e203ca8' --wandb-user 'wqlevi' --image-size 256  --lr-gamma 1 --use-weighting --ckpt 'raven_FM_MDM_AdptWeighting_ConstLR_micro32_corrected_large'
echo "Jobs finished"


