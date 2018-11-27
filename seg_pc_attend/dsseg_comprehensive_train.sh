#!/bin/bash
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128000M        # memory per node
#SBATCH --time=3-00:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load python/3.6.3
source ~/tensorflow/bin/activate
python main.py --pc_size=100000 --semi=True --binary=True --cat=0 > ds_semi_bin_.log
