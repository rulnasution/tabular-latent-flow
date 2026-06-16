#!/bin/bash
#$ -cwd

# #SBATCH -p multicore   # Each job array task will be a multicore job

# #SBATCH -p multicore_small ## this one uses intel
# #SBATCH -m 32          # Run on a 32GB/core (512GB) node (max 16 cores).

# #SBATCH -p gpuL        # Use the GPU partition
# #SBATCH -G 1           # Request 1 GPU

# #SBATCH -p himem       # (or --partition=) A high memory job
# #SBATCH -n 1          # Each job array task will use 4 cores in this example
# #SBATCH -t 4-0         # Each job array task can run for up to 4 days (0 hours)

# This part is if we use the multicore partition

# #SBATCH -p himem       # (or --partition=) A high memory job
#SBATCH -p serial   # Each job array task will be a multicore job

# #SBATCH -p multicore   # Each job array task will be a multicore job

# #SBATCH -p gpuA        # Use the GPU partition
# #SBATCH -G 1           # Request 1 GPU

#SBATCH -n 1           # Each job array task will use 4 cores in this example
#SBATCH -t 7-0         # Each job array task can run for up to 4 days (0 hours)
#SBATCH -a 7-13 # A job-array with 1000 "tasks", numbered 1...1000

# datanames = ['adult','beijing','default','magic','news','shoppers'.
#               'canada', 'fiji', 'uk', 'rwanda', 'indonesia','adulta','churn']
# 
module load apps/binapps/anaconda3/2023.09
source activate tabsyn
module load apps/binapps/pytorch/2.3.0-311-cpu
# module load apps/binapps/pytorch/1.11.0-39-gpu-cu113

# module load apps/binapps/jupyter-notebook/any
# jupyter-notebook-csf -p gpuA -n 12 -g 1 -t 0-16

# module unload modulename
# echo 'trying to run vfm with x0 and x1 prediction. now it is run using 4096 batch size'
echo 'trying to run different variants of flow models'
echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# =CONCAT("--eval_urisk_dir ",G2,"_",LEFT(L2,1),"_",N2,"_",P2)

# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 2 --eval_urisk_dir abc_sde --var_weight velocity_x0 --cond_vel ot --steps 100
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 2 --eval_urisk_dir def_sde --var_weight velocity_x0 --cond_vel ot --t_ode 0.975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 1 --eval_urisk_dir tabsynflow_v_ot_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 2 --eval_urisk_dir tabsynflow_v_ot_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 4 --eval_urisk_dir tabsynflow_v_ot_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 8 --eval_urisk_dir tabsynflow_v_ot_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 16 --eval_urisk_dir tabsynflow_v_ot_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 32 --eval_urisk_dir tabsynflow_v_ot_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 64 --eval_urisk_dir tabsynflow_v_ot_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --eval_urisk_dir tabsynflow_v_ot_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 128 --eval_urisk_dir tabsynflow_v_ot_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 256 --eval_urisk_dir tabsynflow_v_ot_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 512 --eval_urisk_dir tabsynflow_v_ot_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 1024 --eval_urisk_dir tabsynflow_v_ot_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 1 --eval_urisk_dir tabsynflow_v_vp_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 2 --eval_urisk_dir tabsynflow_v_vp_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 4 --eval_urisk_dir tabsynflow_v_vp_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 8 --eval_urisk_dir tabsynflow_v_vp_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 16 --eval_urisk_dir tabsynflow_v_vp_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 32 --eval_urisk_dir tabsynflow_v_vp_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 64 --eval_urisk_dir tabsynflow_v_vp_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --eval_urisk_dir tabsynflow_v_vp_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 128 --eval_urisk_dir tabsynflow_v_vp_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 256 --eval_urisk_dir tabsynflow_v_vp_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 512 --eval_urisk_dir tabsynflow_v_vp_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 1024 --eval_urisk_dir tabsynflow_v_vp_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 1 --eval_urisk_dir tabsynflow_s_ot_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 2 --eval_urisk_dir tabsynflow_s_ot_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 4 --eval_urisk_dir tabsynflow_s_ot_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 8 --eval_urisk_dir tabsynflow_s_ot_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 16 --eval_urisk_dir tabsynflow_s_ot_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 32 --eval_urisk_dir tabsynflow_s_ot_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 64 --eval_urisk_dir tabsynflow_s_ot_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --eval_urisk_dir tabsynflow_s_ot_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 128 --eval_urisk_dir tabsynflow_s_ot_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 256 --eval_urisk_dir tabsynflow_s_ot_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 512 --eval_urisk_dir tabsynflow_s_ot_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 1024 --eval_urisk_dir tabsynflow_s_ot_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 1 --eval_urisk_dir tabsynflow_s_vp_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 2 --eval_urisk_dir tabsynflow_s_vp_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 4 --eval_urisk_dir tabsynflow_s_vp_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 8 --eval_urisk_dir tabsynflow_s_vp_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 16 --eval_urisk_dir tabsynflow_s_vp_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 32 --eval_urisk_dir tabsynflow_s_vp_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 64 --eval_urisk_dir tabsynflow_s_vp_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --eval_urisk_dir tabsynflow_s_vp_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 128 --eval_urisk_dir tabsynflow_s_vp_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 256 --eval_urisk_dir tabsynflow_s_vp_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 512 --eval_urisk_dir tabsynflow_s_vp_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 1024 --eval_urisk_dir tabsynflow_s_vp_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 1 --eval_urisk_dir tabsynflow_d_ot_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 2 --eval_urisk_dir tabsynflow_d_ot_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 4 --eval_urisk_dir tabsynflow_d_ot_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 8 --eval_urisk_dir tabsynflow_d_ot_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 16 --eval_urisk_dir tabsynflow_d_ot_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 32 --eval_urisk_dir tabsynflow_d_ot_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 64 --eval_urisk_dir tabsynflow_d_ot_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --eval_urisk_dir tabsynflow_d_ot_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 128 --eval_urisk_dir tabsynflow_d_ot_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 256 --eval_urisk_dir tabsynflow_d_ot_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 512 --eval_urisk_dir tabsynflow_d_ot_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 1024 --eval_urisk_dir tabsynflow_d_ot_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 1 --eval_urisk_dir tabsynflow_d_vp_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 2 --eval_urisk_dir tabsynflow_d_vp_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 4 --eval_urisk_dir tabsynflow_d_vp_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 8 --eval_urisk_dir tabsynflow_d_vp_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 16 --eval_urisk_dir tabsynflow_d_vp_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 32 --eval_urisk_dir tabsynflow_d_vp_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 64 --eval_urisk_dir tabsynflow_d_vp_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --eval_urisk_dir tabsynflow_d_vp_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 128 --eval_urisk_dir tabsynflow_d_vp_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 256 --eval_urisk_dir tabsynflow_d_vp_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 512 --eval_urisk_dir tabsynflow_d_vp_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 1024 --eval_urisk_dir tabsynflow_d_vp_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 1 --eval_urisk_dir tabsynvlow_e_ot_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 2 --eval_urisk_dir tabsynvlow_e_ot_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 4 --eval_urisk_dir tabsynvlow_e_ot_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 8 --eval_urisk_dir tabsynvlow_e_ot_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 16 --eval_urisk_dir tabsynvlow_e_ot_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 32 --eval_urisk_dir tabsynvlow_e_ot_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 64 --eval_urisk_dir tabsynvlow_e_ot_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --eval_urisk_dir tabsynvlow_e_ot_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 128 --eval_urisk_dir tabsynvlow_e_ot_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 256 --eval_urisk_dir tabsynvlow_e_ot_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 512 --eval_urisk_dir tabsynvlow_e_ot_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 1024 --eval_urisk_dir tabsynvlow_e_ot_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 1 --eval_urisk_dir tabsynvlow_e_vp_1
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 2 --eval_urisk_dir tabsynvlow_e_vp_2
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 4 --eval_urisk_dir tabsynvlow_e_vp_4
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 8 --eval_urisk_dir tabsynvlow_e_vp_8
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 16 --eval_urisk_dir tabsynvlow_e_vp_16
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 32 --eval_urisk_dir tabsynvlow_e_vp_32
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 64 --eval_urisk_dir tabsynvlow_e_vp_64
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --eval_urisk_dir tabsynvlow_e_vp_100
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 128 --eval_urisk_dir tabsynvlow_e_vp_128
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 256 --eval_urisk_dir tabsynvlow_e_vp_256
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 512 --eval_urisk_dir tabsynvlow_e_vp_512
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 1024 --eval_urisk_dir tabsynvlow_e_vp_1024

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --t_ode 0.6 --eval_urisk_dir tabsynflow_v_ot_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --t_ode 0.7 --eval_urisk_dir tabsynflow_v_ot_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --t_ode 0.8 --eval_urisk_dir tabsynflow_v_ot_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --t_ode 0.9 --eval_urisk_dir tabsynflow_v_ot_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --t_ode 0.95 --eval_urisk_dir tabsynflow_v_ot_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --t_ode 0.975 --eval_urisk_dir tabsynflow_v_ot_0975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --t_ode 0.6 --eval_urisk_dir tabsynflow_v_vp_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --t_ode 0.7 --eval_urisk_dir tabsynflow_v_vp_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --t_ode 0.8 --eval_urisk_dir tabsynflow_v_vp_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --t_ode 0.9 --eval_urisk_dir tabsynflow_v_vp_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --t_ode 0.95 --eval_urisk_dir tabsynflow_v_vp_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --t_ode 0.975 --eval_urisk_dir tabsynflow_v_vp_0975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --t_ode 0.6 --eval_urisk_dir tabsynflow_s_ot_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --t_ode 0.7 --eval_urisk_dir tabsynflow_s_ot_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --t_ode 0.8 --eval_urisk_dir tabsynflow_s_ot_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --t_ode 0.9 --eval_urisk_dir tabsynflow_s_ot_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --t_ode 0.95 --eval_urisk_dir tabsynflow_s_ot_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --t_ode 0.975 --eval_urisk_dir tabsynflow_s_ot_0975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --t_ode 0.6 --eval_urisk_dir tabsynflow_s_vp_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --t_ode 0.7 --eval_urisk_dir tabsynflow_s_vp_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --t_ode 0.8 --eval_urisk_dir tabsynflow_s_vp_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --t_ode 0.9 --eval_urisk_dir tabsynflow_s_vp_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --t_ode 0.95 --eval_urisk_dir tabsynflow_s_vp_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --t_ode 0.975 --eval_urisk_dir tabsynflow_s_vp_0975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --t_ode 0.6 --eval_urisk_dir tabsynflow_d_ot_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --t_ode 0.7 --eval_urisk_dir tabsynflow_d_ot_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --t_ode 0.8 --eval_urisk_dir tabsynflow_d_ot_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --t_ode 0.9 --eval_urisk_dir tabsynflow_d_ot_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --t_ode 0.95 --eval_urisk_dir tabsynflow_d_ot_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --t_ode 0.975 --eval_urisk_dir tabsynflow_d_ot_0975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --t_ode 0.6 --eval_urisk_dir tabsynflow_d_vp_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --t_ode 0.7 --eval_urisk_dir tabsynflow_d_vp_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --t_ode 0.8 --eval_urisk_dir tabsynflow_d_vp_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --t_ode 0.9 --eval_urisk_dir tabsynflow_d_vp_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --t_ode 0.95 --eval_urisk_dir tabsynflow_d_vp_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --t_ode 0.975 --eval_urisk_dir tabsynflow_d_vp_0975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --t_ode 0.6 --eval_urisk_dir tabsynvlow_e_ot_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --t_ode 0.7 --eval_urisk_dir tabsynvlow_e_ot_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --t_ode 0.8 --eval_urisk_dir tabsynvlow_e_ot_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --t_ode 0.9 --eval_urisk_dir tabsynvlow_e_ot_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --t_ode 0.95 --eval_urisk_dir tabsynvlow_e_ot_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --t_ode 0.975 --eval_urisk_dir tabsynvlow_e_ot_0975

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --t_ode 0.6 --eval_urisk_dir tabsynvlow_e_vp_06
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --t_ode 0.7 --eval_urisk_dir tabsynvlow_e_vp_07
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --t_ode 0.8 --eval_urisk_dir tabsynvlow_e_vp_08
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --t_ode 0.9 --eval_urisk_dir tabsynvlow_e_vp_09
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --t_ode 0.95 --eval_urisk_dir tabsynvlow_e_vp_095
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --t_ode 0.975 --eval_urisk_dir tabsynvlow_e_vp_0975

echo "Run finishes: $(date '+%d-%m-%Y %H:%M:%S')"