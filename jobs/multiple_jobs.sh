#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00    
#SBATCH --mail-user=<isa45@sfu.ca>
#SBATCH --mail-type=END

BATCH_SIZE=3

cd ../code/
module load python/3.12 scipy-stack
source ~/env_fl/bin/activate

train.py $SLURM_ARRAY_TASK_ID $BATCH_SIZE
