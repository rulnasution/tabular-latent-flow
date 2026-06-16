#!/bin/bash
#$ -cwd

# This part is if we use the multicore partition
# SBATCH -p multicore_small   # Each job array task will be a multicore job
#SBATCH -p himem   # Each job array task will be a multicore job
#SBATCH -n 1           # Each job array task will use 8 cores in this example
#SBATCH -t 4-0         # Each job array task can run for up to 4 days (0 hours)
#SBATCH -a 1-7         # array number 

module load apps/binapps/anaconda3/2023.09
module load apps/binapps/pytorch/1.11.0-39-cpu
source activate synthcity

# country_names = ['Canada','Fiji','UK','Rwanda','Indonesia','Adult','Churn','Insurance','Credit']
echo 'trying to run bayesian_network.'
time python run_benchmarks_latent_flow.py --id $SLURM_ARRAY_TASK_ID --model bayesian_network
