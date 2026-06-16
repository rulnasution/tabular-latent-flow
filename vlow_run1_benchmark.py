import subprocess
import json
import argparse
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

def run_command(command, env_name=None, output=False):
    """ Runs a shell command in a specified Conda environment. """
    if env_name:
        command = f'conda run -n {env_name} {command}'
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=output,text=True)
        if output:
            if result.returncode == 0:
                # Get the last non-empty line of stdout
                last_line = result.stdout.strip().split("\n")[-1]
                return last_line
            else:
                print(f"Error: {result.stderr}")
                return None
        else:
            return result
    except subprocess.CalledProcessError as e:
        print("\n=== COMMAND FAILED ===")
        print("cmd:", e.cmd)
        print("returncode:", e.returncode)

        if e.stdout:
            print("\n--- STDOUT ---\n", e.stdout[:4000])
        if e.stderr:
            print("\n--- STDERR ---\n", e.stderr[:8000])  # ini biasanya ada traceback asli

        raise  # biar tetap fail setelah kita lihat log

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TabVFM')
    # parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--dataname', type=int, default=0, help='Name of dataset.')
    parser.add_argument('--method', type=str, default='tabvfm', help='Algorithm used.')
    parser.add_argument('--n_eval', type=int, default=1, help='number of evaluations.')
    parser.add_argument('--eval_other_dir', type=str, default='', help="directory eval (from tabsyn paper), '' means not evaluated")
    parser.add_argument('--eval_quality_dir', type=str, default='', help='directory eval of quality (from tabsyn paper), '' means not evaluated')
    parser.add_argument('--eval_urisk_dir', type=str, default='', help='directory eval of utility and risk, '' means not evaluated')

    args = parser.parse_args()
    
    datanames = ['adult','beijing','default','magic','news','shoppers', 
                 'canada', 'fiji', 'uk', 'rwanda', 'indonesia','adulta','churn']
    idx = args.dataname-1
    dataname = datanames[idx]
    # print("\n Activating first environment: tabsyn\n")
    
    ## running the model
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    orig_data_dir = f'/mnt/hum01-home01/p88346bn/test/project/bayes-ctgan-fix/benchmarks/results_lfm/'

    if args.eval_quality_dir != '':
        final_res = []
        if os.path.isfile(f"eval/combine_scratch/{dataname}/{args.eval_quality_dir}_quality.csv"):
            print(f'file "eval/combine_scratch/{dataname}/{args.eval_quality_dir}_quality.csv" exists')
        else:
            
            for s in range(args.n_eval):

                syn_file = f'{args.method}_{s}'
                
                run_command(f'python eval/eval_quality_distance.py --dataname {dataname} --model {args.method} --path "{orig_data_dir}/{dataname}/{args.method}/{syn_file}.csv"', env_name="synthcity")    
                with open(f"eval/quality/{dataname}/{args.method}.txt", "r") as file:
                    alpha_pr, beta_re, wd, mmd = map(float, file.read().splitlines())
                print(f"Extracted Values: alpha precision: {alpha_pr}, beta recall: {beta_re}, WD: {wd}, MMD: {wd}")
                # alpha_pr, beta_re = 0, 0
                
                final_keys = ['sample_seed', 'alpha_pr', 'beta_re', 'wd', 'mmd']
                final_values = [s, alpha_pr, beta_re, wd, mmd]
                final_res.append(final_values)
                # print("\n✅ Script finished!")

            final_results = pd.DataFrame(final_res, columns=final_keys)
            save_dir = f'eval/combine_scratch/{dataname}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            final_results.to_csv(f"{save_dir}/{args.eval_quality_dir}_quality.csv", index=False)
            print(f"Quality Results saved to {save_dir}/{args.eval_quality_dir}_quality.csv")

    if args.eval_urisk_dir != '':
        final_res = []
        if os.path.isfile(f"eval/combine_scratch/{dataname}/{args.eval_urisk_dir}_urisk.csv"):
            print(f'file "eval/combine_scratch/{dataname}/{args.eval_urisk_dir}_urisk.csv" exists')
        else:
            for s in range(args.n_eval):
                # print("\n Switching to second environment: synthcity\n")
                
                syn_file = f'{args.method}_{s}'
                
                run_command(f'python eval/eval_urisk.py --dataname {dataname} --model {args.method} --path "{orig_data_dir}/{dataname}/{args.method}/{syn_file}.csv"', env_name="tabsyn")
                with open(f'eval/urisk/{dataname}/{args.method}.json', "r") as file:
                    res_urisk = json.load(file)
    
                run_command(f'python eval/eval_density.py --dataname {dataname} --model {args.method} --path "{orig_data_dir}/{dataname}/{args.method}/{syn_file}.csv"', env_name="tabsyn")
                with open(f"eval/density/{dataname}/{args.method}/quality.txt", "r") as file:
                    shape_val, trend_val = map(float, file.read().splitlines())
                print(f"Extracted Values: Column Shapes: {shape_val}, Column Pair Trends: {trend_val}")    

                # try:
                detection_val = run_command(f'python eval/eval_detection.py --dataname {dataname} --model {args.method} --path "{orig_data_dir}/{dataname}/{args.method}/{syn_file}.csv"', env_name="tabsyn", output=True)
                print(f"Detection Value: {detection_val}")
                # except Exception as e:
                #     print(f'error in detection run {e}')
                #     detection_val = 0
                
                keys = list(res_urisk.keys()) + ["shape", "trend", "detection"]
                extracted_values = list(res_urisk.values()) + [shape_val, trend_val, detection_val]
                
                # keys = list(res_urisk.keys())
                # extracted_values = list(res_urisk.values())
                final_res.append(extracted_values)

            final_results = pd.DataFrame(final_res, columns=keys)
            save_dir = f'eval/combine_scratch/{dataname}'
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            final_results.to_csv(f"{save_dir}/{args.eval_urisk_dir}_urisk.csv", index=False)
            print(f"Final Results saved to {save_dir}/{args.eval_urisk_dir}_urisk.csv")