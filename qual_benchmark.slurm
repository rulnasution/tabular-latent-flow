#!/bin/bash
#$ -cwd

# #SBATCH -p himem       # (or --partition=) A high memory job


#SBATCH -p serial
# #SBATCH -p multicore   # Each job array task will be a multicore job
# #SBATCH -n 3           # Each job array task will use 4 cores in this example

#SBATCH -t 7-0         # Each job array task can run for up to 4 days (0 hours)
#SBATCH -a 7-13 # A job-array with 1000 "tasks", numbered 1...1000

# datanames = ['adult','beijing','default','magic','news','shoppers'.
#               'canada', 'fiji', 'uk', 'rwanda', 'indonesia','adulta','churn']
# 

# ['arf','bayesian_network', 'ctgan','nflow', 'rtvae', 'tvae']

module load apps/binapps/anaconda3/2023.09
module load apps/binapps/pytorch/1.11.0-39-cpu
source activate synthcity
# module load apps/binapps/jupyter-notebook/any
# jupyter-notebook-csf -p gpuA -n 12 -g 1 -t 0-16

# echo 'trying to run adv. random forest evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method arf --n_eval 20 --eval_urisk_dir arf

# echo 'trying to run bayesian network evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method bayesian_network --n_eval 20 --eval_urisk_dir bayesian_network

# echo 'trying to run ctgan evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method ctgan --n_eval 20 --eval_urisk_dir ctgan

# echo 'trying to run normalising flow evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method nflow --n_eval 20 --eval_urisk_dir nflow

# echo 'trying to run tabular vae evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method tvae --n_eval 20 --eval_urisk_dir tvae

# echo 'trying to run robust tabular vae evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method rtvae --n_eval 20 --eval_urisk_dir rtvae



# echo 'trying to run adv. random forest quality evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method arf --n_eval 20 --eval_quality_dir arf

# echo 'trying to run bayesian network quality evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method bayesian_network --n_eval 20 --eval_quality_dir bayesian_network

# echo 'trying to run ctgan quality evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method ctgan --n_eval 20 --eval_quality_dir ctgan

# echo 'trying to run normalising flow quality evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method nflow --n_eval 20 --eval_quality_dir nflow

# echo 'trying to run tabular vae quality evaluation on benchmark models'
# echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method tvae --n_eval 20 --eval_quality_dir tvae

echo 'trying to run robust tabular vae quality evaluation on benchmark models'
echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

time python vlow_run1_benchmark.py --dataname ${SLURM_ARRAY_TASK_ID} --method rtvae --n_eval 20 --eval_quality_dir rtvae


echo "Run finishes: $(date '+%d-%m-%Y %H:%M:%S')"