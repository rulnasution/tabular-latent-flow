import numpy as np
import pandas as pd
import os 

import json

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import statsmodels.api as sm
from sklearn.preprocessing import KBinsDiscretizer

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='tabsyn')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


def eval_syn_data(name, orig, syn):
    country_name = name.upper() if name=='uk' else name.capitalize()
    country_name = 'Adult' if name == 'adulta' else country_name
    roc_val = cal_mean_roc(country_name,orig,syn)
    cio_val = cal_mean_cio(country_name,orig,syn)
    if country_name not in ['Adult', 'Churn']:
        orig['AGE'] = orig['AGE'].astype(str)
        syn['AGE'] = syn['AGE'].astype(str)
    tcap_val = cal_mean_tcap(country_name,orig,syn)
    evaluation = {'ROC_uni': roc_val[0], 'ROC_biv': roc_val[1], 'CIO': cio_val, 'TCAP': tcap_val, 
    'Utility':(roc_val[0]+roc_val[1]+cio_val)/3, 'Risk': max(0, tcap_val)}
    # evaluation = [roc_val[0], roc_val[1], cio_val, tcap_val, (roc_val[0]+roc_val[1]+cio_val)/3, max(0, tcap_val)]
    return evaluation

def cal_mean_cio(name,orig,syn):
    if name == 'UK':
        target_cols = ['TENURE','MSTATUS']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS','TENURE','MSTATUS']
        # get columns used and make y to be binary
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MSTATUS'] = (orig['MSTATUS'] == 'Married' ) | (orig['MSTATUS'] == 'Remarried' )
        orig['TENURE'] = (orig['TENURE'] == 'Own occ-buying' ) | (orig['TENURE'] == 'Own occ-outright' )
        syn['MSTATUS'] = (syn['MSTATUS'] == 'Married' ) | (syn['MSTATUS'] == 'Remarried' )
        syn['TENURE'] = (syn['TENURE'] == 'Own occ-buying' ) | (syn['TENURE'] == 'Own occ-outright' )
    elif name=='Canada':
        target_cols = ['TENURE','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['ABIDENT','CLASSWK','DEGREE','EMPSTAT','SEX','URBAN','TENURE','MARST']
        # get columns used and make y to be binary
        cont_cols = ['AGE', 'HRSWK', 'INCTOT', 'WKSWORK']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a4' )| (orig['MARST'] == '2' ) | (orig['MARST'] == '4' )).astype('int')
        orig['TENURE'] =((orig['TENURE'] == 'a1' ) | (orig['TENURE'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a4' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '4' )).astype('int')
        syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
    elif name=='Fiji':
        target_cols = ['TENURE','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['CLASSWKR','ETHNIC','RELIGION','EDATTAIN','SEX','PROV','TENURE','MARST']
        # get columns used and make y to be binary
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' )|(orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
        orig['TENURE'] =((orig['TENURE'] == 'a1' )|(orig['TENURE'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
        syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
    elif name=='Rwanda':
        target_cols = ['OWNERSH','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['DISAB1','EDCERT','CLASSWK','LIT','RELIG','SEX','OWNERSH','MARST']
        # get columns used and make y to be binary
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
        orig['OWNERSH'] =((orig['OWNERSH'] == 'a1' ) | (orig['OWNERSH'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
        syn['OWNERSH'] = ((syn['OWNERSH'] == 'a1' ) | (syn['OWNERSH'] == '1' )).astype('int')
    elif name=='Indonesia':
        target_cols = ['OWNERSHIP','MARST']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['LANDOWN', 'RELATE', 'SEX', 'HOMEFEM', 'HOMEMALE', 'RELIGION', 
                    'LIT', 'SCHOOL', 'EDATTAIND', 'DISABLED','OWNERSHIP','MARST']
        cont_cols = ['AGE']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]
        
        orig['MARST'] = ((orig['MARST'] == '3' ) | (orig['MARST'] == '4' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == 'a4' )).astype('int')
        orig['OWNERSHIP'] =((orig['OWNERSHIP'] == 'a1' ) | (orig['OWNERSHIP'] == '1' )).astype('int')
        syn['MARST'] = ((syn['MARST'] == '3' ) | (syn['MARST'] == '4' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == 'a4' )).astype('int')
        syn['OWNERSHIP'] = ((syn['OWNERSHIP'] == 'a1' ) | (syn['OWNERSHIP'] == '1' )).astype('int')
    elif name=='Adult':
        target_cols = ['income','marital-status']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['workclass', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country','income']
        cont_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['income'] = ((orig['income'] == '>50K' ) | (orig['income'] == '>50K.' )).astype('int')
        orig['marital-status'] =((orig['marital-status'] == 'Married-civ-spouse' ) | (orig['marital-status'] == 'Married-spouse-absent' ) | (orig['marital-status'] == 'Married-AF-spouse')).astype('int')
        syn['income'] = ((syn['income'] == '>50K' ) | (syn['income'] == '>50K.' )).astype('int')
        syn['marital-status'] =((syn['marital-status'] == 'Married-civ-spouse' ) | (syn['marital-status'] == 'Married-spouse-absent' ) | (syn['marital-status'] == 'Married-AF-spouse')).astype('int')
    elif name=='Churn':
        target_cols = ['Exited','CreditScore','EstimatedSalary']
        families = [sm.families.Binomial(),sm.families.Gaussian(),sm.families.Gaussian()]
        key_cols = ['Geography', 'Gender', 'Tenure', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
        cont_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]
        
        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['Exited'] = ((orig['Exited'] == '1' )).astype('int')
        syn['Exited'] = ((syn['Exited'] == '1' )).astype('int')
    elif name=='Insurance':
        target_cols = ['charges']
        families = [sm.families.Gaussian()]
        
        key_cols = ['sex', 'children', 'smoker', 'region']
        cont_cols = ['charges','age','bmi']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]

        orig = orig[key_cols]
        syn = syn[key_cols]
        # median_cutoff = orig['charges'].median()
        # orig['charges'] = ((orig['charges'] > median_cutoff )).astype('int')
        # syn['charges'] = ((syn['charges'] > median_cutoff )).astype('int')
    elif name=='Credit':
        target_cols = ['checking_balance', 'default']
        families = [sm.families.Binomial(),sm.families.Binomial()]
        key_cols = ['checking_balance', 'credit_history', 'purpose',
                    'savings_balance', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']
        cont_cols = ['months_loan_duration', 'amount', 'age']
        orig_cont = orig[cont_cols]
        syn_cont = syn[cont_cols]

        orig = orig[key_cols]
        syn = syn[key_cols]

        orig['checking_balance'] = ((orig['checking_balance'] == '1 - 200 DM') | (orig['checking_balance'] == '> 200 DM')).astype('int')
        orig['credit_history'] =((orig['credit_history'] == 'repaid') | (orig['credit_history'] == 'fully repaid') | (orig['credit_history'] == 'fully repaid this bank')).astype('int')
        orig['savings_balance'] = ((orig['savings_balance'] == '501 - 1000 DM' ) | (orig['savings_balance'] == '> 1000 DM')).astype('int')
        syn['checking_balance'] = ((syn['checking_balance'] == '1 - 200 DM') | (syn['checking_balance'] == '> 200 DM')).astype('int')
        syn['credit_history'] =((syn['credit_history'] == 'repaid' ) | (syn['credit_history'] == 'fully repaid') | (syn['credit_history'] == 'fully repaid this bank')).astype('int')
        syn['savings_balance'] =((syn['savings_balance'] == '1 - 200 DM') | (syn['savings_balance'] == '> 1000 DM') ).astype('int')

    orig.fillna('a0',inplace=True)
    syn.fillna('a0',inplace=True)
    encoder = OrdinalEncoder()
    encoder.fit(pd.concat([orig.astype(str),syn.astype(str)],axis=0))
    orig = pd.DataFrame(encoder.transform(orig.astype(str)),columns=key_cols)
    syn = pd.DataFrame(encoder.transform(syn.astype(str)),columns=key_cols)
    if len(cont_cols) > 0:
        orig = pd.concat([orig,orig_cont],axis=1)
        syn = pd.concat([syn,syn_cont],axis=1)
        key_cols += cont_cols

    scores = []
    for target, family in zip(target_cols, families):
        orig_glm = sm.GLM(orig[target].astype(float),orig.drop(columns=target).astype(float),family=family)
        syn_glm = sm.GLM(syn[target].astype(float),syn.drop(columns=target).astype(float),family=family)
        results = CIO_function(orig_glm, syn_glm)
        scores.append(results['mean_ci_overlap_noNeg'])
    return np.mean(scores)

# def cal_mean_cio(name,orig,syn):
#     if name == 'UK':
#         target_cols = ['TENURE','MSTATUS']
#         key_cols = ['AGE','ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS','TENURE','MSTATUS']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MSTATUS'] = (orig['MSTATUS'] == 'Married' ) | (orig['MSTATUS'] == 'Remarried' )
#         orig['TENURE'] = (orig['TENURE'] == 'Own occ-buying' ) | (orig['TENURE'] == 'Own occ-outright' )
#         syn['MSTATUS'] = (syn['MSTATUS'] == 'Married' ) | (syn['MSTATUS'] == 'Remarried' )
#         syn['TENURE'] = (syn['TENURE'] == 'Own occ-buying' ) | (syn['TENURE'] == 'Own occ-outright' )
#     elif name=='Canada':
#         target_cols = ['TENURE','MARST']
#         key_cols = ['ABIDENT','AGE','CLASSWK','DEGREE','EMPSTAT','SEX','URBAN','TENURE','MARST']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a4' )| (orig['MARST'] == '2' ) | (orig['MARST'] == '4' )).astype('int')
#         orig['TENURE'] =((orig['TENURE'] == 'a1' ) | (orig['TENURE'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a4' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '4' )).astype('int')
#         syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
#     elif name=='Fiji':
#         target_cols = ['TENURE','MARST']
#         key_cols = ['AGE','CLASSWKR','ETHNIC','RELIGION','EDATTAIN','SEX','PROV','TENURE','MARST']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' )|(orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
#         orig['TENURE'] =((orig['TENURE'] == 'a1' )|(orig['TENURE'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
#         syn['TENURE'] = ((syn['TENURE'] == 'a1' ) | (syn['TENURE'] == '1' )).astype('int')
#     elif name=='Rwanda':
#         target_cols = ['OWNERSH','MARST']
#         key_cols = ['AGE','DISAB1','EDCERT','CLASSWK','LIT','RELIG','SEX','OWNERSH','MARST']
#         # get columns used and make y to be binary
#         orig = orig[key_cols]
#         syn = syn[key_cols]
#         orig['MARST'] = ((orig['MARST'] == 'a2' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == '2' ) | (orig['MARST'] == '3' )).astype('int')
#         orig['OWNERSH'] =((orig['OWNERSH'] == 'a1' ) | (orig['OWNERSH'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == 'a2' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == '2' ) | (syn['MARST'] == '3' )).astype('int')
#         syn['OWNERSH'] = ((syn['OWNERSH'] == 'a1' ) | (syn['OWNERSH'] == '1' )).astype('int')
#     elif name=='Indonesia':
#         target_cols = ['OWNERSHIP','MARST']
#         key_cols = ['LANDOWN', 'AGE', 'RELATE', 'SEX', 'HOMEFEM', 'HOMEMALE', 'RELIGION', 
#                     'LIT', 'SCHOOL', 'EDATTAIND', 'DISABLED','OWNERSHIP','MARST']
#         orig = orig[key_cols]
#         syn = syn[key_cols]
        
#         orig['MARST'] = ((orig['MARST'] == '3' ) | (orig['MARST'] == '4' ) | (orig['MARST'] == 'a3' ) | (orig['MARST'] == 'a4' )).astype('int')
#         orig['OWNERSHIP'] =((orig['OWNERSHIP'] == 'a1' ) | (orig['OWNERSHIP'] == '1' )).astype('int')
#         syn['MARST'] = ((syn['MARST'] == '3' ) | (syn['MARST'] == '4' ) | (syn['MARST'] == 'a3' ) | (syn['MARST'] == 'a4' )).astype('int')
#         syn['OWNERSHIP'] = ((syn['OWNERSHIP'] == 'a1' ) | (syn['OWNERSHIP'] == '1' )).astype('int')
#     elif name=='Adult':
#         target_cols = ['income','marital-status']
#         key_cols = ['workclass', 'education-num',
#                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                     'native-country','income']
#         ## 'age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['income'] = ((orig['income'] == '>50K' ) | (orig['income'] == '>50K.' )).astype('int')
#         orig['marital-status'] =((orig['marital-status'] == 'Married-civ-spouse' ) | (orig['marital-status'] == 'Married-spouse-absent' ) | (orig['marital-status'] == 'Married-AF-spouse')).astype('int')
#         syn['income'] = ((syn['income'] == '>50K' ) | (syn['income'] == '>50K.' )).astype('int')
#         syn['marital-status'] =((syn['marital-status'] == 'Married-civ-spouse' ) | (syn['marital-status'] == 'Married-spouse-absent' ) | (syn['marital-status'] == 'Married-AF-spouse')).astype('int')
#     elif name=='Churn':
#         target_cols = ['Exited']
#         key_cols = ['Geography', 'Gender', 'Tenure', 
#                     'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
#         ## 'CreditScore', 'Age', 'Balance', 'EstimatedSalary'
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['Exited'] = ((orig['Exited'] == '1' )).astype('int')
#         syn['Exited'] = ((syn['Exited'] == '1' )).astype('int')
#     elif name=='Insurance':
#         target_cols = ['charges']
#         key_cols = ['sex', 'children', 'smoker', 'region', 'charges']
#         ## 'age','bmi'
#         median_cutoff = orig['charges'].median()
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['charges'] = ((orig['charges'] > median_cutoff )).astype('int')
#         syn['charges'] = ((syn['charges'] > median_cutoff )).astype('int')
#     elif name=='Credit':
#         target_cols = ['checking_balance', 'default']
#         key_cols = ['checking_balance', 'credit_history', 'purpose',
#                     'savings_balance', 'employment_length', 'installment_rate',
#                     'personal_status', 'other_debtors', 'residence_history', 'property',
#                     'installment_plan', 'housing', 'existing_credits', 'default',
#                     'dependents', 'telephone', 'foreign_worker', 'job']
#         ## 'months_loan_duration', 'amount', 'age'
#         orig = orig[key_cols]
#         syn = syn[key_cols]

#         orig['checking_balance'] = ((orig['checking_balance'] == '1 - 200 DM') | (orig['checking_balance'] == '> 200 DM')).astype('int')
#         orig['credit_history'] =((orig['credit_history'] == 'repaid') | (orig['credit_history'] == 'fully repaid') | (orig['credit_history'] == 'fully repaid this bank')).astype('int')
#         orig['savings_balance'] = ((orig['savings_balance'] == '501 - 1000 DM' ) | (orig['savings_balance'] == '> 1000 DM')).astype('int')
#         syn['checking_balance'] = ((syn['checking_balance'] == '1 - 200 DM') | (syn['checking_balance'] == '> 200 DM')).astype('int')
#         syn['credit_history'] =((syn['credit_history'] == 'repaid' ) | (syn['credit_history'] == 'fully repaid') | (syn['credit_history'] == 'fully repaid this bank')).astype('int')
#         syn['savings_balance'] =((syn['savings_balance'] == '1 - 200 DM') | (syn['savings_balance'] == '> 1000 DM') ).astype('int')

#     orig.fillna('a0',inplace=True)
#     syn.fillna('a0',inplace=True)
#     encoder = OrdinalEncoder()
#     encoder.fit(pd.concat([orig.astype(str),syn.astype(str)],axis=0))
#     orig = pd.DataFrame(encoder.transform(orig.astype(str)),columns=key_cols)
#     syn = pd.DataFrame(encoder.transform(syn.astype(str)),columns=key_cols)
#     scores = []
#     for target in target_cols:
#         orig_glm = sm.GLM(orig[target].astype(float),orig.drop(columns=target).astype(float),family=sm.families.Binomial() )
#         syn_glm = sm.GLM(syn[target].astype(float),syn.drop(columns=target).astype(float),family=sm.families.Binomial() )
#         results = CIO_function(orig_glm, syn_glm)
#         scores.append(results['mean_ci_overlap_noNeg'])
#     return np.mean(scores)

def CIO_function(orig_glm,syn_glm):
    # # put them into a form so it is easier to extract the coefficients etc.
    try:
        syn_glm = syn_glm.fit()
        orig_glm = orig_glm.fit()
    except:
        return {'mean_std_coef_diff':0, 
                'median_std_coef_diff' : 0,
                'mean_ci_overlap':0, 
                'median_ci_overlap' : 0,
                'mean_ci_overlap_noNeg' :0, 
                'median_ci_overlap_noNeg':0}  # when there is a perfect separation in syn dataset

    syn_glm = pd.DataFrame(syn_glm.summary().tables[1].data[1:],columns=['names','Estimate','stderr','z','P>|z|','[0.25','0.975]'])
    orig_glm = pd.DataFrame(orig_glm.summary().tables[1].data[1:],columns=['names','Estimate','stderr','z','P>|z|','[0.25','0.975]'])
    syn_glm = syn_glm.iloc[:,:3] # take the first three columns
    orig_glm = orig_glm.iloc[:,:3]
    
    # join the original and synth
    combined = orig_glm.merge(syn_glm,how='left',on='names',suffixes=('_orig', '_syn'))
    for i in combined.columns[1:]:
        combined[i] = combined[i].astype('float')
    combined['std.coef_diff'] = abs(combined['Estimate_orig']-combined['Estimate_syn']) / (combined['stderr_orig'])
    combined['orig_lower'] = combined['Estimate_orig'] - 1.96 * combined['stderr_orig']
    combined['orig_upper'] = combined['Estimate_orig'] + 1.96 * combined['stderr_orig']
    combined['syn_lower'] = combined['Estimate_syn'] - 1.96 * combined['stderr_syn']
    combined['syn_upper'] = combined['Estimate_syn'] + 1.96 * combined['stderr_syn']
    combined['ci_overlap'] = 0.5 * (
                                    (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
                                    (combined['orig_upper']-combined['orig_lower']) + 
                                    (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
                                    (combined['syn_upper']-combined['syn_lower'])
                                    )
    for index,row in combined.iterrows():
        if row['orig_lower'] == row['orig_upper'] and row['orig_upper'] == row['syn_lower'] and row['syn_upper'] == row['syn_lower']:
            combined.loc[index,'ci_overlap'] = 1.0
    combined = combined[['names','std.coef_diff','ci_overlap']]
    
    combined.fillna(0,inplace=True) # set negative overlaps to zero
    combined['ci_overlap_noNeg'] = [0 if i<0 else i for i in combined['ci_overlap']]

    results = {'mean_std_coef_diff':combined['std.coef_diff'].mean(), 
                'median_std_coef_diff' : combined['std.coef_diff'].median(),
                'mean_ci_overlap': combined.ci_overlap.mean(), 
                'median_ci_overlap' : combined.ci_overlap.median(),
                # add in the overlaps where negatives were changed to zeros
                'mean_ci_overlap_noNeg' :combined.ci_overlap_noNeg.mean(), 
                'median_ci_overlap_noNeg':combined.ci_overlap_noNeg.median()}
    # now compute std. diff and ci overlap
    return results

# def CIO_function_cont(orig,syn):
#     # # CIO for continuous variables
#     # # make consistent form with CIO_function
#     try:
#         n = len(orig)
#         combined = pd.DataFrame({'Estimate_orig':orig.mean(),
#                         'Estimate_syn':syn.mean(),
#                         'stderr_orig':orig.std()/(n**0.5),
#                         'stderr_syn':syn.std()/(n**0.5),
#                         }).reset_index()
#         combined.rename(columns={'index': 'names'}, inplace=True)
#     except:
#         return {'mean_std_coef_diff':0, 
#                 'median_std_coef_diff' : 0,
#                 'mean_ci_overlap':0, 
#                 'median_ci_overlap' : 0,
#                 'mean_ci_overlap_noNeg' :0, 
#                 'median_ci_overlap_noNeg':0}  # when there is a perfect separation in syn dataset

#     combined['std.coef_diff'] = abs(combined['Estimate_orig']-combined['Estimate_syn']) / (combined['stderr_orig'])
#     combined['orig_lower'] = combined['Estimate_orig'] - 1.96 * combined['stderr_orig']
#     combined['orig_upper'] = combined['Estimate_orig'] + 1.96 * combined['stderr_orig']
#     combined['syn_lower'] = combined['Estimate_syn'] - 1.96 * combined['stderr_syn']
#     combined['syn_upper'] = combined['Estimate_syn'] + 1.96 * combined['stderr_syn']
#     combined['ci_overlap'] = 0.5 * (
#                                     (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
#                                     (combined['orig_upper']-combined['orig_lower']) + 
#                                     (combined[['orig_upper','syn_upper']].min(axis=1) - combined[['orig_lower','syn_lower']].max(axis=1)) /
#                                     (combined['syn_upper']-combined['syn_lower'])
#                                     )
#     print(combined)
#     for index,row in combined.iterrows():
#         if row['orig_lower'] == row['orig_upper'] and row['orig_upper'] == row['syn_lower'] and row['syn_upper'] == row['syn_lower']:
#             combined.loc[index,'ci_overlap'] = 1.0
#     combined = combined[['names','std.coef_diff','ci_overlap']]
    
#     combined.fillna(0,inplace=True) # set negative overlaps to zero
#     combined['ci_overlap_noNeg'] = [0 if i<0 else i for i in combined['ci_overlap']]

#     results = {'mean_std_coef_diff':combined['std.coef_diff'].mean(), 
#                 'median_std_coef_diff' : combined['std.coef_diff'].median(),
#                 'mean_ci_overlap': combined.ci_overlap.mean(), 
#                 'median_ci_overlap' : combined.ci_overlap.median(),
#                 # add in the overlaps where negatives were changed to zeros
#                 'mean_ci_overlap_noNeg' :combined.ci_overlap_noNeg.mean(), 
#                 'median_ci_overlap_noNeg':combined.ci_overlap_noNeg.median()}
#     # now compute std. diff and ci overlap
#     return results


def cal_mean_roc(name,orig,syn,bi=True):
    if name == 'UK':
        key_cols = ['ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS',
                    'TENURE','MSTATUS']
    elif name=='Canada':
        key_cols = ['ABIDENT','SEX','TENURE','URBAN','BPLMOM','BPLPOP',
                    'CITIZEN','LANG','MARST','RELATE','MINORITY','RELIG','BPL']
    elif name=='Fiji':
        key_cols = ['PROV','TENURE','RELATE','SEX','ETHNIC','MARST',
                    'RELIGION','BPLPROV','RESPROV',
                    'RESSTAT','SCHOOL','TRAVEL']
    elif name=='Rwanda':
        key_cols = ['STATUS','SEX','URBAN','OWNERSH','DISAB2','DISAB1',
                    'RELATE','RELIG','HINS','NATION','BPL']
    elif name=='Indonesia':
        key_cols = ['OWNERSHIP', 'LANDOWN', 'RELATE', 'SEX', 'MARST', 
                    'HOMEMALE', 'RELIGION', 'SCHOOL', 'LIT', 'EDATTAIND', 'DISABLED']
    elif name=='Adult':
        key_cols = ['workclass', 'education',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country','income']
    elif name=='Churn':
        key_cols = ['Geography', 'Gender', 'Tenure', 
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
    elif name=='Insurance':
        key_cols = ['sex', 'children', 'smoker', 'region', 'charges']
    elif name=='Credit':
        key_cols = ['checking_balance', 'credit_history', 'purpose',
                    'savings_balance', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']
    orig = orig[key_cols]
    syn = syn[key_cols]


    # if name == 'UK':
    #     key_cols = ['ECONPRIM','ETHGROUP','LTILL','QUALNUM','SEX','SOCLASS',
    #                 'TENURE','MSTATUS']
    #     num_cols = ['AGE','HOURS']
    # elif name=='Canada':
    #     key_cols = ['ABIDENT','SEX','TENURE','URBAN','BPLMOM','BPLPOP',
    #                 'CITIZEN','LANG','MARST','RELATE','MINORITY','RELIG','BPL']
    #     num_cols = ['AGE','HRSWK','INCTOT','WKSWORK']
    # elif name=='Fiji':
    #     key_cols = ['PROV','TENURE','RELATE','SEX','ETHNIC','MARST',
    #                 'RELIGION','BPLPROV','RESPROV',
    #                 'RESSTAT','SCHOOL','TRAVEL']
    #     num_cols = ['AGE']
    # elif name=='Rwanda':
    #     key_cols = ['STATUS','SEX','URBAN','OWNERSH','DISAB2','DISAB1',
    #                 'RELATE','RELIG','HINS','NATION','BPL']
    #     num_cols = ['AGE']
    # elif name=='Indonesia':
    #     key_cols = ['OWNERSHIP', 'LANDOWN', 'RELATE', 'SEX', 'MARST', 
    #                 'HOMEMALE', 'RELIGION', 'SCHOOL', 'LIT', 'EDATTAIND', 'DISABLED']
    #     num_cols = ['AGE']
    # elif name=='Adult':
    #     key_cols = ['workclass', 'education',
    #                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
    #                 'native-country','income']
    #     num_cols = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
    # elif name=='Churn':
    #     key_cols = ['Geography', 'Gender', 'Tenure', 
    #                 'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']
    #     num_cols = ['CreditScore', 'Age', 'Balance','EstimatedSalary']
    # elif name=='Insurance':
    #     key_cols = ['sex', 'children', 'smoker', 'region']
    #     num_cols = ['age','bmi','charges']
    # elif name=='Credit':
    #     key_cols = ['checking_balance', 'credit_history', 'purpose',
    #                 'savings_balance', 'employment_length', 'installment_rate',
    #                 'personal_status', 'other_debtors', 'residence_history', 'property',
    #                 'installment_plan', 'housing', 'existing_credits', 'default',
    #                 'dependents', 'telephone', 'foreign_worker', 'job']
    #     num_cols = ['months_loan_duration','amount','age']

    # kb = KBinsDiscretizer(encode='ordinal')
    # kb.fit(orig[num_cols])
    # orig[num_cols] = kb.transform(orig[num_cols]).astype(int)
    # syn[num_cols] = kb.transform(syn[num_cols]).astype(int)
    # orig = orig[key_cols+num_cols]
    # syn = syn[key_cols+num_cols]
    
    uni_scores = []
    bi_scores = []
    for i in range(len(key_cols)):
        uni_scores.append(roc_univariate(orig, syn, i) )
      
        if bi and i+1<len(key_cols):# max i == len(key_cols)-1
            for j in range(i+1,len(key_cols)):
                bi_scores.append(roc_bivariate(orig, syn, i, j))
    if bi:
        return np.mean(uni_scores),np.mean(bi_scores)
    else:
        return np.mean(uni_scores),0

def roc_univariate(original,synthetic,var_num):
    # create frequency tables for the original and synthetic data, on the variable
    orig_table = original.iloc[:,var_num].value_counts().reset_index()
    syn_table = synthetic.iloc[:,var_num].value_counts().reset_index()
    orig_table.columns = ['value','Freq']
    syn_table.columns = ['value','Freq']
    # calculate the proportions by dividing by the number of records in each dataset
    orig_table['prop'] = orig_table.Freq/len(original)
    syn_table['prop'] = syn_table.Freq/len(synthetic)
    # merge the two tables, by the variable
    combined = orig_table.merge(syn_table,on=['value'],how='outer')
    # merging will induce NAs where there is a category mismatch - i.e. the category exists in one dataset but not the other
    # to deal with this set the NA values to zero:
    combined.fillna(0,inplace=True)
    # get the maximum proportion for each category level:
    combined['max'] = combined[['prop_x','prop_y']].max(axis=1)
    # get the minimum proportion for each category level:
    combined['min'] = combined[['prop_x','prop_y']].min(axis=1)
    # roc is min divided by max (a zero value for min results in a zero for ROC, as expected)
    combined['roc'] = combined['min'] / combined['max']
    combined['roc'].fillna(1,inplace=True)
    return combined['roc'].mean()


def roc_bivariate(original, synthetic, var1, var2):
    # create frequency tables for the original and synthetic data, on the two variable cross-tabulation
    orig_table = pd.crosstab(index=original.iloc[:,var1],columns=original.iloc[:,var2]).stack().reset_index()
    syn_table = pd.crosstab(index=synthetic.iloc[:,var1],columns=synthetic.iloc[:,var2]).stack().reset_index()
    orig_table.columns = ['Var1','Var2','Freq']
    syn_table.columns = ['Var1','Var2','Freq']
    # calculate the proportions by dividing by the number of records in each dataset
    orig_table['prop'] = orig_table.Freq/len(original)
    syn_table['prop'] = syn_table.Freq/len(synthetic)
    # merge the two tables, by the variables
    combined = orig_table.merge(syn_table,on=['Var1', 'Var2'],how='outer')
    # merging will induce NAs where there is a category mismatch - i.e. the category exists in one dataset but not the other
    # to deal with this set the NA values to zero:
    combined.fillna(0,inplace=True)
    # get the maximum proportion for each category level:
    combined['max'] = combined[['prop_x','prop_y']].max(axis=1)
    # get the minimum proportion for each category level:
    combined['min'] = combined[['prop_x','prop_y']].min(axis=1)
    # roc is min divided by max (a zero value for min results in a zero for ROC, as expected)
    combined['roc'] = combined['min'] / combined['max']
    combined['roc'].fillna(1,inplace=True)
    return combined['roc'].mean()


'''
function:     replace_missing()   
description:  replaces missing values dependant on data type. Categorical or object NAs are replaced with 'blank', numerical NAs with -999. Can be modified as required
input:        pandas dataframe
output:       pandas dataframe with missing values replaced
'''
def replace_missing(dataset):
    # get a dictionary of the different data types
    types = dataset.dtypes.to_dict()
    # replace object or categorical NAs with 'blank', and numerical with -999
    for col_nam, typ in types.items():
        if (typ == 'O' or typ == 'c'):
            dataset[col_nam] = dataset[col_nam].fillna('blank')
        if (typ == 'float64' or typ == 'int64'):
            dataset[col_nam] = dataset[col_nam].fillna(-999)
    return(dataset)

def cal_mean_tcap(name,orig,syn):
    if name=='UK':
        target_cols = ['LTILL','FAMTYPE','TENURE']
        key_cols = ['AREAP','AGE','SEX','MSTATUS','ETHGROUP','ECONPRIM']
    elif name=='Canada':
        target_cols = ['RELIG','CITIZEN','TENURE']
        key_cols = ['AGE','SEX','MARST','MINORITY','EMPSTAT','BPL']
    elif name=='Fiji':
        target_cols = ['RELIGION','WORKTYPE','TENURE']
        key_cols = ['PROV','AGE','SEX','MARST','ETHNIC','CLASSWKR']
    elif name=='Rwanda':
        target_cols = ['RELIG','WKSECTOR','OWNERSH']
        key_cols = ['AGE','SEX','MARST','CLASSWK','URBAN','BPL']
    elif name=='Indonesia':
        target_cols = ['RELIGION','OWNERSHIP','EDATTAIND']
        key_cols = ['AGE', 'SEX', 'MARST', 'HOMEFEM','SCHOOL', 'LANDOWN']
    elif name=='Adult':
        target_cols = ['native-country', 'race', 'occupation']
        key_cols = ['education-num', 'marital-status', 'workclass', 
                    'relationship', 'sex', 'income']
    elif name=='Churn':
        target_cols = ['Geography']
        key_cols = ['Gender', 'Tenure', 'NumOfProducts', 
                    'HasCrCard', 'IsActiveMember','Exited']
    elif name=='Insurance':
        target_cols = ['children']
        key_cols = ['sex','region', 'smoker']
    elif name=='Credit':
        target_cols = ['checking_balance', 'credit_history', 'savings_balance']
        key_cols = ['purpose', 'employment_length', 'installment_rate',
                    'personal_status', 'other_debtors', 'residence_history', 'property',
                    'installment_plan', 'housing', 'existing_credits', 'default',
                    'dependents', 'telephone', 'foreign_worker', 'job']

    # if name=='UK':
    #     target_cols = ['LTILL','FAMTYPE','TENURE']
    #     key_cols = ['AREAP','AGE','HOURS','SEX','MSTATUS','ETHGROUP','ECONPRIM']
    #     num_cols = ['AGE','HOURS']
    # elif name=='Canada':
    #     target_cols = ['RELIG','CITIZEN','TENURE']
    #     key_cols = ['AGE','SEX','MARST','MINORITY','EMPSTAT','BPL',
    #                 'HRSWK','INCTOT','WKSWORK']
    #     num_cols = ['AGE','HRSWK','INCTOT','WKSWORK']
    # elif name=='Fiji':
    #     target_cols = ['RELIGION','WORKTYPE','TENURE']
    #     key_cols = ['PROV','AGE','SEX','MARST','ETHNIC','CLASSWKR']
    #     num_cols = ['AGE']
    # elif name=='Rwanda':
    #     target_cols = ['RELIG','WKSECTOR','OWNERSH']
    #     key_cols = ['AGE','SEX','MARST','CLASSWK','URBAN','BPL']
    #     num_cols = ['AGE']
    # elif name=='Indonesia':
    #     target_cols = ['RELIGION','OWNERSHIP','EDATTAIND']
    #     key_cols = ['AGE', 'SEX', 'MARST', 'HOMEFEM','SCHOOL', 'LANDOWN']
    #     num_cols = ['AGE']
    # elif name=='Adult':
    #     target_cols = ['native-country', 'race', 'occupation']
    #     key_cols = ['education-num', 'marital-status', 'workclass', 
    #                 'relationship', 'sex', 'income',
    #                 'age','fnlwgt','capital-gain','capital-loss','hours-per-week'
    #                 ]
    #     num_cols = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
    # elif name=='Churn':
    #     target_cols = ['Geography']
    #     key_cols = ['Gender', 'Tenure', 'NumOfProducts', 
    #                 'HasCrCard', 'IsActiveMember','Exited',
    #                 'CreditScore', 'Age', 'Balance','EstimatedSalary']
    #     num_cols = ['CreditScore', 'Age', 'Balance','EstimatedSalary']
    # elif name=='Insurance':
    #     target_cols = ['children','charges']
    #     key_cols = ['sex','region', 'smoker','age','bmi']
    #     num_cols = ['age','bmi','charges']
    # elif name=='Credit':
    #     target_cols = ['checking_balance', 'credit_history', 'savings_balance']
    #     key_cols = ['purpose', 'employment_length', 'installment_rate',
    #                 'personal_status', 'other_debtors', 'residence_history', 'property',
    #                 'installment_plan', 'housing', 'existing_credits', 'default',
    #                 'dependents', 'telephone', 'foreign_worker', 'job',
    #                 'months_loan_duration','amount','age']
    #     num_cols = ['months_loan_duration','amount','age']

    # kb = KBinsDiscretizer(n_bins=3,encode='ordinal')
    # kb.fit(orig[num_cols])
    # orig[num_cols] = kb.transform(orig[num_cols]).astype(int)
    # syn[num_cols] = kb.transform(syn[num_cols]).astype(int)
    
    scores = []
    for target in target_cols:
        for i in range(3,len(key_cols)+1):
            score,baseline = tcap(orig,syn,target,key_cols[:i],verbose=False)
            # print(score,baseline)
            score_scaled = (score - baseline)/(1-baseline)
            scores.append(score_scaled)
    return np.mean(scores)


'''
function:     tcap()   
description:  takes the original and synthetic dataset filenames and a set of keys/target variables and calculates the TCAP score
input:        original = location/filename of original dataset
              synth = location/filename of synthetic dataset
              num_keys = number of key variables
              target = target variable
              key = key variable as the baseline 
              verbose = if set to True it will print out more detailed results
output:       TCAP score and the baseline value for that target variable
'''
def tcap(orig, syn, target, key, verbose=False):
       
    # read in the data
    #orig = pd.read_csv(original)
    #syn = pd.read_csv(synth)
    
    # define the keys and target. using the num_keys parameter means that a dataset with any number of columns can
    # be used, and only the relevant keys analysed
    keys_target = key + [target]
    num_keys = len(key)
    # print(keys_target)
    
    # select just the required columns (keys and target)    
    orig = orig[keys_target]
    syn = syn[keys_target]
    # replace any missing values
    orig = replace_missing(orig)
    syn = replace_missing(syn)
    # count the categories for the target (for calculating baseline)
    uvd = orig[target].value_counts()
    
    # use groupby to get the equivalance classes for synthetic data
    eqkt_syn = pd.DataFrame({'count' : syn.groupby( keys_target ).size()}).reset_index()           # with target
    eqk_syn = pd.DataFrame({'count' : syn.groupby( keys_target[:num_keys] ).size()}).reset_index() # without target
    # equivalance classes for original data without target
    eqk_orig = pd.DataFrame({'count' : orig.groupby( keys_target[:num_keys] ).size()}).reset_index()

    # merge with original to calculate baseline    
    orig_merge_eqk = pd.merge(orig, eqk_orig, on= keys_target[:num_keys]) 
    orig_merge_eqk.rename({'count': 'count_eqk_orig'}, axis=1, inplace=True)
    # calculate the baseline
    uvt = sum(uvd[orig_merge_eqk[target]]/sum(uvd))
    baseline = uvt/len(orig)
    
    # calculate synthetic cap score. merge syn eq classes (with keys) with syn eq classes (with keys/target)
    syn_merge = eqk_syn.merge(eqkt_syn, on=keys_target[:num_keys])
    syn_merge['prop'] = syn_merge['count_y']/syn_merge['count_x']
    # filter out those less than tau=1
    syn_merge = syn_merge[syn_merge['prop'] >= 1]
    # merge with original, if in syn eq classes (just keys) then this is a matching record (Taub)
    syn_merge = syn_merge.merge(orig_merge_eqk, on=keys_target[:num_keys], how='inner')
    matching_records = len(syn_merge)

    # drop records where the targets are not equal
    syn_merge = syn_merge[syn_merge[target + '_x']==syn_merge[target + '_y']]
    dcaptotal = len(syn_merge)

    if matching_records == 0:
        tcap_undef = 0
    else:
        tcap_undef = dcaptotal/matching_records
   
    # output is [the TCAP as used by Taub, and the baseline]. Modify as required
    output = ([tcap_undef,baseline])
    
    if verbose==True:
        print('TCAP calculation')
        print('===============')
        print('Source dataset is: ',orig)
        print('Target dataset is: ',syn)
        print('The total number of records in the source dataset is: ', len(orig))
        print('The total number of records in the target dataset is: ', len(syn))
        print('The target variable is: ', target)
        print('The key size is: ', num_keys)
        print('The keys are: ', key)
        print('Number of matching records: ', matching_records)
        print('DCAP total is: ', dcaptotal)
        print('TCAP with non-matches undefined is: ', tcap_undef)
        print('The baseline is: ', baseline)

    return(output)

def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]
    real_cat_data = real_cat_data.replace('nan', np.nan, inplace=True)
    
    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    # new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    syn_cat_data = syn_cat_data.replace('nan', np.nan, inplace=True)
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    # new_syn_data.columns = range(len(new_syn_data.columns))
    
    metadata = info['metadata']

    # columns = metadata['columns']
    metadata['columns'] = {}

    # inverse_idx_mapping = info['inverse_idx_mapping']


    # for i in range(len(new_real_data.columns)):
    #     if i < len(num_col_idx):
    #         metadata['columns'][i] = columns[num_col_idx[i]]
    #     else:
    #         metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]

    return new_real_data, new_syn_data, metadata

if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'

    data_dir = f'data/{dataname}' 
    # print(syn_path)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path, dtype=str)
    real_data = pd.read_csv(real_path, dtype=str)

    discrete_columns = [real_data.columns[i] for i in info['cat_col_idx']]
    numerical_columns = [real_data.columns[i] for i in info['num_col_idx']]
    if info['task_type'] == 'binclass': discrete_columns += real_data.columns[info['target_col_idx']].tolist()
    else: numerical_columns += real_data.columns[info['target_col_idx']].tolist()

    # print(real_data.values)
    # print(real_data.info())

    # print(syn_data.head())
    # print(syn_data.values)
    
    real_data[numerical_columns] = real_data[numerical_columns].astype(float)
    if dataname not in ['adulta', 'churn']:
        real_data['AGE'] = real_data['AGE'].round(0).astype(int)
    
    # real_data[discrete_columns] = real_data[discrete_columns].replace('nan', np.nan, inplace=True)
    real_data[discrete_columns] = real_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                      value=r'\1', regex=True)
    
    syn_data[numerical_columns] = syn_data[numerical_columns].astype(float)
    if dataname not in ['adulta', 'churn']:
        syn_data['AGE'] = syn_data['AGE'].round(0).astype(int)
    # syn_data[discrete_columns] = syn_data[discrete_columns].replace('nan', np.nan, inplace=True)
    syn_data[discrete_columns] = syn_data[discrete_columns].replace(to_replace=r'(\d+)\.0\b', 
                                                                      value=r'\1', regex=True)
    
    # print(real_data.head())
    # print(syn_data.head())
    # new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    results = eval_syn_data(dataname, real_data, syn_data)
    
    save_dir = f'eval/urisk/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f'eval/urisk/{dataname}/{model}.json'
    print('Saving scores to ', save_path)

    with open(save_path, "w") as json_file:
        json.dump(results, json_file, indent=4, separators=(", ", ": "))