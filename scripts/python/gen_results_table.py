import json
import sys
import os

exp_name = sys.argv[1]
dataset_name = sys.argv[2]
results_dir = 'exps/{}/best/results'.format(exp_name)

results_all = {}
for filename in os.listdir(results_dir):
    if filename.endswith('_results.json'):
        with open(os.path.join(results_dir, filename), 'r') as f:
            results = json.load(f)
        expl_name = filename.split('_')[0]
        results_all[expl_name] = results

expl_names_all = {
    'lime': 'LIME',
    'shap': 'SHAP',
    'rise': 'RISE',
    'gradcam': 'Grad-CAM',
    'intgrad': 'IntGrad',
    'archipelago': 'Archi-pelago',
    'idg': 'IDG',
    'pls': 'PLS',
    'fresh': 'FRESH',
    'sop': 'SOP (ours)'
}
# expl_names_list = [k for k in ['lime', 'shap', 'rise', 'gradcam', 'intgrad', 'archipelago', 'fresh', 'sop'] if k in results_all]
if dataset_name in ['imagenet']:
    expl_names_list = ['lime', 'shap', 'rise', 'gradcam', 'intgrad', 'archipelago', 'fresh', 'sop']
else:
    expl_names_list = ['lime', 'shap', 'rise', 'intgrad', 'archipelago', 'idg', 'pls', 'fresh', 'sop']
# expl_names = {k:expl_names_all[k] for k in results_all if k in results_all}

dataset_names = {
    'imagenet': 'ImageNet',
    'multirc': 'MultiRC',
    'sst': 'SST'
}

def gen_latex_table(results_all, expl_names, expl_names_list, dataset_name, dataset_names):
    print('\\toprule')
    print(' & & ', end='')
    for expl_name in expl_names_list:
        print('{} & '.format(expl_names[expl_name]), end='')
    print('\\\\ \\midrule')
    print('\\multirow{3}{*}{' + dataset_names[dataset_name] + '} & Accuracy $\\uparrow$ ', end='')
    accuracy_mean_max = max([results_all[expl_name]['accuracy_mean'] for expl_name in expl_names_list if expl_name in results_all])
    for expl_name in expl_names_list:
        if expl_name in results_all:
            if results_all[expl_name]['accuracy_mean'] == accuracy_mean_max:
                print('& \\textbf{{{:.4f}}} '.format(results_all[expl_name]['accuracy_mean']), end='')
            else:
                print('& {:.4f} '.format(results_all[expl_name]['accuracy_mean']), end='')
        else:
            print(' & ', end='')
    print('\\\\')
    print(' & Consistency $\\uparrow$ ', end='')
    consistency_mean_max = max([results_all[expl_name]['consistency_mean'] for expl_name in expl_names_list if expl_name in results_all])
    for expl_name in expl_names_list:
        if expl_name in results_all:
            if results_all[expl_name]['consistency_mean'] == consistency_mean_max:
                print('& \\textbf{{{:.4f}}} '.format(results_all[expl_name]['consistency_mean']), end='')
            else:
                print('& {:.4f} '.format(results_all[expl_name]['consistency_mean']), end='')
        else:
            print(' & ', end='')
    print('\\\\')
    print(' & KL-divergence $\\downarrow$ ', end='')
    kl_div_mean_min = min([results_all[expl_name]['kl_div_mean'] for expl_name in expl_names_list if expl_name in results_all])
    for expl_name in expl_names_list:
        if expl_name in results_all:
            if results_all[expl_name]['kl_div_mean'] == kl_div_mean_min:
                print('& \\textbf{{{:.4f}}} '.format(results_all[expl_name]['kl_div_mean']), end='')
            else:
                print('& {:.4f} '.format(results_all[expl_name]['kl_div_mean']), end='')
        else:
            print(' & ', end='')
    print('\\\\')
    print('\\bottomrule')

gen_latex_table(results_all, expl_names_all, expl_names_list, dataset_name, dataset_names)