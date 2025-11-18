#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=0:15:0    
#SBATCH --mail-user=<isa45@sfu.ca>
#SBATCH --mail-type=ALL

cd ../code/
module load python/3.12 scipy-stack
source ~/env_fl/bin/activate

python preprocessing.py
