#!/bin/bash
#$ -cwd

#SBATCH -p gpuA        # Use the GPU partition
#SBATCH -G 1           # Request 1 GPU
#SBATCH -n 1           # Each job array task will use 1 core in this example
#SBATCH -t 4-0         # Each job array task can run for up to 4 days (0 hours)
#SBATCH -a 5         # array number

module load apps/binapps/anaconda3/2023.09
module load apps/binapps/pytorch/1.11.0-39-gpu-cu113
source activate synthcity

# country_names = ['Canada','Fiji','UK','Rwanda','Indonesia','Adult','Churn','Insurance','Credit']

# echo 'trying to run tvae.'
# time python run_benchmarks_latent_flow.py --id $SLURM_ARRAY_TASK_ID --model tvae

# echo 'trying to run rtvae.'
# time python run_benchmarks_latent_flow.py --id $SLURM_ARRAY_TASK_ID --model rtvae

echo 'trying to run ctgan.'
time python run_benchmarks_latent_flow.py --id $SLURM_ARRAY_TASK_ID --model ctgan

echo 'trying to run normalizing flow.'
time python run_benchmarks_latent_flow.py --id $SLURM_ARRAY_TASK_ID --model nflow

# echo 'trying to run tddpm.'
# time python run_benchmarks_latent_flow.py --id $SLURM_ARRAY_TASK_ID --model ddpm