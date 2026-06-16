import os
import argparse
from pathlib import Path
import pandas as pd

from sklearn.datasets import load_iris

## Imputation module
import xgboost
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

## Main module
from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file, load_from_file
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(os.getcwd())

dfs = [] 
# data_dir = '/mnt/hum01-home01/p88346bn/test/project/tab-ddpm/'
data_dir = '/mnt/hum01-home01/p88346bn/test/project/benchmarks/tabsyn/data/'
country_names = ['canada','fiji','uk','rwanda','indonesia','adulta','churn']
for i in range(7):
    dfs.append(pd.read_csv(f"{data_dir}/{country_names[i]}/train.csv", dtype=str))
    print(country_names[i])

cont_cols_canada = ['HRSWK','INCTOT','WKSWORK']
cat_cols_canada = [i for i in dfs[0].columns if i not in cont_cols_canada]
dfs[0][cont_cols_canada] = dfs[0][cont_cols_canada].astype(float)

cont_col_adult = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
cont_col_churn = ['CreditScore', 'Age', 'Balance','EstimatedSalary']

## adult

cont_col_noncensus = [cont_col_adult, cont_col_churn]
cont_columns = [cont_cols_canada,['AGE'],['AGE'],['AGE'],['AGE']] + cont_col_noncensus
discrete_col = [[j for j in dfs[i].columns if j not in cont_columns[i]] for i in range(7)]

# discrete_col_noncensus = [[j for j in dfs[i].columns if j not in cont_col_noncensus[i-5]] for i in range(5,9)]

# discrete_columns = [cat_cols_canada,dfs[1].columns.tolist(),dfs[2].columns.tolist(),
#                     dfs[3].columns.tolist(),dfs[4].columns.tolist()] + discrete_col_noncensus


for i in range(7):
    dfs[i][cont_columns[i]] = dfs[i][cont_columns[i]].astype(float)
    
y_cols = ['TENURE','TENURE','TENURE','MARST','MARST',
          'income','Exited','charges','checking_balance']

param_dict = {'ctgan': {'generator_n_layers_hidden': 3, 
                        'generator_n_units_hidden': 640,
                        'discriminator_n_layers_hidden': 3, 
                        'discriminator_n_units_hidden': 640},
            'tvae': {'encoder_n_layers_hidden': 3, 
                    'encoder_n_units_hidden': 768,
                    'decoder_n_layers_hidden': 3, 
                    'decoder_n_units_hidden': 768
                    },
            'rtvae': {'encoder_n_layers_hidden': 3, 
                    'encoder_n_units_hidden': 768,
                    'decoder_n_layers_hidden': 3, 
                    'decoder_n_units_hidden': 768
                    },
            'nflow': {'n_layers_hidden': 2,
                    'n_units_hidden': 1280}
             }


parser = argparse.ArgumentParser(description='Latent flow Benchmarking')
# parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--id', type=int, default=1, help='Name of dataset.')
parser.add_argument('--model', type=str, default='bayesian_network', help='Algorithm used.')

args = parser.parse_args()

def main(args):
    ## their default n_iter is 1000
    # Details are in 
    # https://synthcity.readthedocs.io/en/latest/generators.html#general-purpose
    # batch_size = 500 ## following ctgan
    # lr = 2e-3
    # our lr in gan = 2e-4
    # NF in Fiji only possible until 120 epochs, 150 epochs = collapsed 

    print(f"Running {args.model} on {country_names[args.id - 1]} dataset.")

    epochs = 10000
    if args.model in ["bayesian_network"]:
        plugin = Plugins().get(args.model)
    elif args.model in ["ctgan","tvae", "rtvae", "nflow"]:
        plugin = Plugins().get(args.model, n_iter = epochs, device=device,
                               batch_size = 4096, **param_dict[args.model])
    
    plugin.fit(dfs[args.id - 1])

    save_path = Path(f"results_lfm/{country_names[args.id - 1]}/{args.model}/model.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_to_file(save_path, plugin)
    print("Model Saved to:", save_path)

    for i in range(20):
        # generate again without fit ulang
        data_syn = plugin.generate(count=len(dfs[args.id - 1])).dataframe()
        data_syn.to_csv(f"results_lfm/{country_names[args.id - 1]}/{args.model}/{args.model}_{i}.csv", index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)