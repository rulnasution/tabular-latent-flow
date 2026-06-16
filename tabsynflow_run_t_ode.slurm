#!/bin/bash
#$ -cwd

# This part is if we use the multicore partition
# SBATCH -p multicore   # Each job array task will be a multicore job

#SBATCH -p gpuL        # Use the GPU partition
#SBATCH -G 1           # Request 1 GPU
#SBATCH -n 4           # Each job array task will use 4 cores in this example
#SBATCH -t 4-0         # Each job array task can run for up to 4 days (0 hours)
#SBATCH -a 10 # A job-array with 1000 "tasks", numbered 1...1000

# datanames = ['adult','beijing','default','magic','news','shoppers'.
#               'canada', 'fiji', 'uk', 'rwanda', 'indonesia','adulta','churn']
# 
module load apps/binapps/anaconda3/2023.09
source activate tabsyn
module load apps/binapps/pytorch/2.3.0-311-gpu-cu121
# module load apps/binapps/pytorch/1.11.0-39-gpu-cu113

# module load apps/binapps/jupyter-notebook/any
# jupyter-notebook-csf -p gpuA -n 12 -g 1 -t 0-16

# module unload modulename
# echo 'trying to run vfm with x0 and x1 prediction. now it is run using 4096 batch size'
echo 'trying to run different variants of flow models'
echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_06 --var_weight velocity_x0 --cond_vel ot --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_07 --var_weight velocity_x0 --cond_vel ot --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_08 --var_weight velocity_x0 --cond_vel ot --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_09 --var_weight velocity_x0 --cond_vel ot --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_095 --var_weight velocity_x0 --cond_vel ot --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_0975 --var_weight velocity_x0 --cond_vel ot --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_06 --var_weight velocity_x0 --cond_vel vp --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_07 --var_weight velocity_x0 --cond_vel vp --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_08 --var_weight velocity_x0 --cond_vel vp --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_09 --var_weight velocity_x0 --cond_vel vp --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_095 --var_weight velocity_x0 --cond_vel vp --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_0975 --var_weight velocity_x0 --cond_vel vp --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_ot_06 --var_weight score --cond_vel ot --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_ot_07 --var_weight score --cond_vel ot --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_ot_08 --var_weight score --cond_vel ot --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_ot_09 --var_weight score --cond_vel ot --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_ot_095 --var_weight score --cond_vel ot --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_ot_0975 --var_weight score --cond_vel ot --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_06 --var_weight score --cond_vel vp --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_07 --var_weight score --cond_vel vp --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_08 --var_weight score --cond_vel vp --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_09 --var_weight score --cond_vel vp --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_095 --var_weight score --cond_vel vp --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_0975 --var_weight score --cond_vel vp --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_ot_06 --var_weight ddpm --cond_vel ot --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_ot_07 --var_weight ddpm --cond_vel ot --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_ot_08 --var_weight ddpm --cond_vel ot --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_ot_09 --var_weight ddpm --cond_vel ot --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_ot_095 --var_weight ddpm --cond_vel ot --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_ot_0975 --var_weight ddpm --cond_vel ot --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_06 --var_weight ddpm --cond_vel vp --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_07 --var_weight ddpm --cond_vel vp --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_08 --var_weight ddpm --cond_vel vp --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_09 --var_weight ddpm --cond_vel vp --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_095 --var_weight ddpm --cond_vel vp --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_0975 --var_weight ddpm --cond_vel vp --t_ode 0.975


time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_06 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_07 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_08 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_09 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_095 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_0975 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_06 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_07 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_08 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_09 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_095 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_0975 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_06 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_07 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_08 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_09 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_095 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_0975 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.975

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_06 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.6
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_07 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.7
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_08 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_09 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.9
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_095 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.95
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_0975 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --t_ode 0.975


time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_1 --var_weight velocity_x0 --cond_vel ot --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_2 --var_weight velocity_x0 --cond_vel ot --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_4 --var_weight velocity_x0 --cond_vel ot --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_8 --var_weight velocity_x0 --cond_vel ot --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_16 --var_weight velocity_x0 --cond_vel ot --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_32 --var_weight velocity_x0 --cond_vel ot --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_64 --var_weight velocity_x0 --cond_vel ot --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_128 --var_weight velocity_x0 --cond_vel ot --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_256 --var_weight velocity_x0 --cond_vel ot --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_512 --var_weight velocity_x0 --cond_vel ot --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_ot_1024 --var_weight velocity_x0 --cond_vel ot --steps 1024

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_1 --var_weight velocity_x0 --cond_vel vp --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_2 --var_weight velocity_x0 --cond_vel vp --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_4 --var_weight velocity_x0 --cond_vel vp --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_8 --var_weight velocity_x0 --cond_vel vp --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_16 --var_weight velocity_x0 --cond_vel vp --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_32 --var_weight velocity_x0 --cond_vel vp --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_64 --var_weight velocity_x0 --cond_vel vp --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_128 --var_weight velocity_x0 --cond_vel vp --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_256 --var_weight velocity_x0 --cond_vel vp --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_512 --var_weight velocity_x0 --cond_vel vp --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_vp_1024 --var_weight velocity_x0 --cond_vel vp --steps 1024

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_1 --var_weight score --cond_vel vp --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_2 --var_weight score --cond_vel vp --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_4 --var_weight score --cond_vel vp --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_8 --var_weight score --cond_vel vp --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_16 --var_weight score --cond_vel vp --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_32 --var_weight score --cond_vel vp --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_64 --var_weight score --cond_vel vp --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_128 --var_weight score --cond_vel vp --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_256 --var_weight score --cond_vel vp --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_512 --var_weight score --cond_vel vp --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_vp_1024 --var_weight score --cond_vel vp --steps 1024

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_1 --var_weight ddpm --cond_vel vp --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_2 --var_weight ddpm --cond_vel vp --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_4 --var_weight ddpm --cond_vel vp --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_8 --var_weight ddpm --cond_vel vp --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_16 --var_weight ddpm --cond_vel vp --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_32 --var_weight ddpm --cond_vel vp --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_64 --var_weight ddpm --cond_vel vp --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_128 --var_weight ddpm --cond_vel vp --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_256 --var_weight ddpm --cond_vel vp --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_512 --var_weight ddpm --cond_vel vp --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_vp_1024 --var_weight ddpm --cond_vel vp --steps 1024


time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_1 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_2 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_4 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_8 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_16 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_32 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_64 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_128 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_256 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_512 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_ot_vp_1024 --var_weight velocity_x0 --cond_vel ot --sde --cond_vel_sigma vp --steps 1024

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_1 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_2 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_4 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_8 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_16 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_32 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_64 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_128 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_256 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_512 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_v_sde_vp_ot_1024 --var_weight velocity_x0 --cond_vel vp --sde --cond_vel_sigma ot --steps 1024

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_1 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_2 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_4 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_8 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_16 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_32 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_64 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_128 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_256 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_512 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_s_sde_vp_ot_1024 --var_weight score --cond_vel vp --sde --cond_vel_sigma ot --steps 1024

time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_1 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 1
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_2 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 2
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_4 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 4
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_8 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 8
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_16 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 16
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_32 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 32
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_64 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 64
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_128 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 128
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_256 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 256
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_512 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 512
time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --sample --eval_urisk_dir tabsynflow_d_sde_vp_ot_1024 --var_weight ddpm --cond_vel vp --sde --cond_vel_sigma ot --steps 1024



echo "Run finishes: $(date '+%d-%m-%Y %H:%M:%S')"