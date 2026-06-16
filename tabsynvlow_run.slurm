#!/bin/bash
#$ -cwd

# This part is if we use the multicore partition
# SBATCH -p multicore   # Each job array task will be a multicore job

#SBATCH -p gpuL        # Use the GPU partition
#SBATCH -G 1           # Request 1 GPU
#SBATCH -n 4           # Each job array task will use 4 cores in this example
#SBATCH -t 4-0         # Each job array task can run for up to 4 days (0 hours)
#SBATCH -a 7-13 # A job-array with 1000 "tasks", numbered 1...1000

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
echo 'trying to run different integration on ode and sde'
echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabfm --n_eval 20 --train --sample --eval_urisk_dir atabfm_ot_euler --cond_vel ot --t_ode 1
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_ot_euler --cond_vel ot --t_ode 1 --int_method euler
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_ot_midpoint --cond_vel ot --t_ode 1 --int_method midpoint
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_ot_dopri5 --cond_vel ot --t_ode 1 --int_method dopri5
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_ot_rk4 --cond_vel ot --t_ode 1 --int_method rk4
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_vp_sde_ot_euler --cond_vel vp --sde --cond_vel_sigma ot --sigma_max 1.0 --int_method euler
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_vp_sde_ot_heun --cond_vel vp --sde --cond_vel_sigma ot --sigma_max 1.0 --int_method heun
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_vp_sde_ot_midpoint --cond_vel vp --sde --cond_vel_sigma ot --sigma_max 1.0 --int_method midpoint
# time python vlow_run1.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir atabsynvlow_vp_sde_ot_milstein --cond_vel vp --sde --cond_vel_sigma ot --sigma_max 1.0 --int_method milstein




# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel ot --eval_urisk_dir tabsynvlow_ot
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel vp --eval_urisk_dir tabsynvlow_vp
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel ve --eval_urisk_dir tabsynvlow_ve
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel cos --eval_urisk_dir tabsynvlow_cos
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel logit --eval_urisk_dir tabsynvlow_logit

# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel ot --eval_urisk_dir tabsynvlow_ot
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel vp --eval_urisk_dir tabsynvlow_vp
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel ve --eval_urisk_dir tabsynvlow_ve
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel cos --eval_urisk_dir tabsynvlow_cos
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --train --sample --cond_vel logit --eval_urisk_dir tabsynvlow_logit


# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_06 --cond_vel ot --t_ode 0.6
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_06 --cond_vel vp --t_ode 0.6
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_07 --cond_vel ot --t_ode 0.7
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_07 --cond_vel vp --t_ode 0.7
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_08 --cond_vel ot --t_ode 0.8
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_08 --cond_vel vp --t_ode 0.8
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_09 --cond_vel ot --t_ode 0.9
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_09 --cond_vel vp --t_ode 0.9
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_095 --cond_vel ot --t_ode 0.95
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_095 --cond_vel vp --t_ode 0.95
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_0975 --cond_vel ot --t_ode 0.975
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_0975 --cond_vel vp --t_ode 0.975

# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_1 --cond_vel ot --steps 1
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_2 --cond_vel ot --steps 2
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_4 --cond_vel ot --steps 4
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_8 --cond_vel ot --steps 8
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_16 --cond_vel ot --steps 16
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_32 --cond_vel ot --steps 32
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_64 --cond_vel ot --steps 64
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_128 --cond_vel ot --steps 128
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_256 --cond_vel ot --steps 256
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_512 --cond_vel ot --steps 512
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_1024 --cond_vel ot --steps 1024

# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_1 --cond_vel vp --steps 1
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_2 --cond_vel vp --steps 2
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_4 --cond_vel vp --steps 4
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_8 --cond_vel vp --steps 8
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_16 --cond_vel vp --steps 16
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_32 --cond_vel vp --steps 32
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_64 --cond_vel vp --steps 64
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_128 --cond_vel vp --steps 128
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_256 --cond_vel vp --steps 256
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_512 --cond_vel vp --steps 512
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_1024 --cond_vel vp --steps 1024

# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_sde_ot --cond_vel ot --sde --cond_vel_sigma ot --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_sde_vp --cond_vel ot --sde --cond_vel_sigma vp --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_sde_ve --cond_vel ot --sde --cond_vel_sigma ve --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_sde_cos --cond_vel ot --sde --cond_vel_sigma cos --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_ot_sde_logit --cond_vel ot --sde --cond_vel_sigma logit --sigma_max 1.0

# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_sde_ot --cond_vel vp --sde --cond_vel_sigma ot --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_sde_vp --cond_vel vp --sde --cond_vel_sigma vp --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_sde_ve --cond_vel vp --sde --cond_vel_sigma ve --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_sde_cos --cond_vel vp --sde --cond_vel_sigma cos --sigma_max 1.0
# time python flow_run.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --sample --eval_urisk_dir tabsynvlow_vp_sde_logit --cond_vel vp --sde --cond_vel_sigma logit --sigma_max 1.0

echo "Run finishes: $(date '+%d-%m-%Y %H:%M:%S')"