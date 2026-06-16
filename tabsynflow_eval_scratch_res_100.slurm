#!/bin/bash
#$ -cwd

# This part is if we use the multicore partition
# SBATCH -p multicore  # Each job array task will be a multicore job

#SBATCH -p serial   # Each job array task will be a multicore job
#SBATCH -n 1           # Each job array task will use 4 cores in this example
#SBATCH -t 7-0         # Each job array task can run for up to 4 days (0 hours)
#SBATCH -a 7-13 # A job-array with 1000 "tasks", numbered 1...1000

# datanames = ['adult','beijing','default','magic','news','shoppers'.
#        'canada', 'fiji', 'uk', 'rwanda', 'indonesia','adulta','churn']
# 

module load apps/binapps/anaconda3/2023.09
module load apps/binapps/pytorch/1.11.0-39-cpu
source activate synthcity

# module load apps/binapps/jupyter-notebook/any
# jupyter-notebook-csf -p gpuA -n 12 -g 1 -t 0-16

# module unload modulename
# echo 'trying to run vfm with x0 and x1 prediction. now it is run using 4096 batch size'
echo 'trying to run different variants of flow models'
echo "Run started: $(date '+%d-%m-%Y %H:%M:%S')"

# time python vlow_run1_scratch22.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 1 --var_weight velocity_x0 --cond_vel ot --steps 100 --int_method euler --eval_urisk_dir abc


# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --int_method euler --eval_urisk_dir tabsynflow_v_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --int_method euler --eval_urisk_dir tabsynflow_v_ode_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --int_method euler --eval_urisk_dir tabsynflow_s_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --int_method euler --eval_urisk_dir tabsynflow_s_ode_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --int_method euler --eval_urisk_dir tabsynflow_d_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --int_method euler --eval_urisk_dir tabsynflow_d_ode_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --int_method euler --eval_urisk_dir tabsynvlow_e_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --int_method euler --eval_urisk_dir tabsynvlow_e_ode_vp_euler

# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --int_method midpoint --eval_urisk_dir tabsynflow_v_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --int_method midpoint --eval_urisk_dir tabsynflow_v_ode_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --int_method midpoint --eval_urisk_dir tabsynflow_s_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --int_method midpoint --eval_urisk_dir tabsynflow_s_ode_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --int_method midpoint --eval_urisk_dir tabsynflow_d_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --int_method midpoint --eval_urisk_dir tabsynflow_d_ode_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --int_method midpoint --eval_urisk_dir tabsynvlow_e_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --int_method midpoint --eval_urisk_dir tabsynvlow_e_ode_vp_midpoint

# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --sde --int_method euler --eval_urisk_dir tabsynflow_v_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --sde --int_method euler --eval_urisk_dir tabsynflow_v_sde_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --sde --int_method euler --eval_urisk_dir tabsynflow_s_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --sde --int_method euler --eval_urisk_dir tabsynflow_s_sde_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --sde --int_method euler --eval_urisk_dir tabsynflow_d_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --sde --int_method euler --eval_urisk_dir tabsynflow_d_sde_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --sde --int_method euler --eval_urisk_dir tabsynvlow_e_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --sde --int_method euler --eval_urisk_dir tabsynvlow_e_sde_vp_euler

# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynflow_v_sde_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynflow_v_sde_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynflow_s_sde_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynflow_s_sde_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynflow_d_sde_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynflow_d_sde_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynvlow_e_sde_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --sde --int_method midpoint --eval_urisk_dir tabsynvlow_e_sde_vp_midpoint





# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --int_method euler --eval_quality_dir tabsynflow_v_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --int_method euler --eval_quality_dir tabsynflow_v_ode_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --int_method euler --eval_quality_dir tabsynflow_s_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --int_method euler --eval_quality_dir tabsynflow_s_ode_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --int_method euler --eval_quality_dir tabsynflow_d_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --int_method euler --eval_quality_dir tabsynflow_d_ode_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --int_method euler --eval_quality_dir tabsynvlow_e_ode_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --int_method euler --eval_quality_dir tabsynvlow_e_ode_vp_euler

# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --int_method midpoint --eval_quality_dir tabsynflow_v_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --int_method midpoint --eval_quality_dir tabsynflow_v_ode_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --int_method midpoint --eval_quality_dir tabsynflow_s_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --int_method midpoint --eval_quality_dir tabsynflow_s_ode_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --int_method midpoint --eval_quality_dir tabsynflow_d_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --int_method midpoint --eval_quality_dir tabsynflow_d_ode_vp_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --int_method midpoint --eval_quality_dir tabsynvlow_e_ode_ot_midpoint
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --int_method midpoint --eval_quality_dir tabsynvlow_e_ode_vp_midpoint

# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --sde --int_method euler --eval_quality_dir tabsynflow_v_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --sde --int_method euler --eval_quality_dir tabsynflow_v_sde_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --sde --int_method euler --eval_quality_dir tabsynflow_s_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --sde --int_method euler --eval_quality_dir tabsynflow_s_sde_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --sde --int_method euler --eval_quality_dir tabsynflow_d_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --sde --int_method euler --eval_quality_dir tabsynflow_d_sde_vp_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --sde --int_method euler --eval_quality_dir tabsynvlow_e_sde_ot_euler
# time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --sde --int_method euler --eval_quality_dir tabsynvlow_e_sde_vp_euler

time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel ot --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynflow_v_sde_ot_midpoint
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight velocity_x0 --cond_vel vp --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynflow_v_sde_vp_midpoint
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel ot --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynflow_s_sde_ot_midpoint
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight score --cond_vel vp --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynflow_s_sde_vp_midpoint
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel ot --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynflow_d_sde_ot_midpoint
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynflow --n_eval 20 --var_weight ddpm --cond_vel vp --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynflow_d_sde_vp_midpoint
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel ot --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynvlow_e_sde_ot_midpoint
time python vlow_run1_scratch2.py --dataname ${SLURM_ARRAY_TASK_ID} --method tabsynvlow --n_eval 20 --var_weight efvfm --cond_vel vp --steps 100 --sde --int_method midpoint --eval_quality_dir tabsynvlow_e_sde_vp_midpoint


echo "Run finishes: $(date '+%d-%m-%Y %H:%M:%S')"