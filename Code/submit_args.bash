#!/bin/bash
sbatch -A kite_gpu <<EOT
#!/bin/bash

#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -t 3-00:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=38
#SBATCH --job-name=sbatch_EgoAHFA_classification
#SBATCH -o /cluster/home/t63164uhn/Code/EgoAHFA/slurm_logs/%x-%j.out 

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

source /cluster/home/t63164uhn/.bashrc
conda activate /cluster/home/t63164uhn/miniconda3/envs/pt1
cd /cluster/home/t63164uhn/Code/EgoAHFA/
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
python /cluster/home/t63164uhn/Code/EgoAHFA/Code/slowfast_main.py $@
