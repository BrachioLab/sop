import os
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
import random
from exlib.evaluators.common import convert_idx_masks_to_bool

# import sys
# sys.path.append('..')
# sys.path.append('../../lib/exlib/src')
# from sop_utils import *

def sum_weights_for_unique_masks(masks, masks_weights, logits): #, poolers):
    # Convert each boolean mask to a unique string of 0s and 1s
    mask_strs = [''.join(map(str, mask.bool().int().flatten().tolist())) for mask in masks]
    img_size = 66

    # Dictionary to store summed weights for each unique mask
    unique_masks_weights = {}
    unique_masks_logits = {}
    unique_masks_count = {}
    unique_masks_dict = {}

    for i, (mask_str, weight, pred) in enumerate(zip(mask_strs, masks_weights, logits)):
        if mask_str in unique_masks_weights:
            unique_masks_weights[mask_str] += weight
            unique_masks_logits[mask_str] += pred
            unique_masks_count[mask_str] += 1
        else:
            unique_masks_dict[mask_str] = masks[i]
            unique_masks_weights[mask_str] = weight
            unique_masks_logits[mask_str] = pred
            unique_masks_count[mask_str] = 1

    # Convert dictionary keys back to boolean masks
    unique_keys = sorted(unique_masks_weights.keys())
    unique_masks = [unique_masks_dict[key] for key in unique_keys]
    summed_weights = [unique_masks_weights[key] for key in unique_keys]
    mean_logits = [unique_masks_logits[key] for key in unique_keys]

    return unique_masks, summed_weights, mean_logits #, mean_poolers



def get_masks_from_mask(attribution, explainer_name, model=None, k=0.2):
    device = attribution.device
    cell_size = 11
    image_size = 66
    # import pdb; pdb.set_trace()
    mask = torch.arange(1, cell_size*cell_size + 1, dtype=torch.int).reshape(cell_size, cell_size)

    # Resize the mask to (224, 224) without using intermediate floating point numbers
    # This can be achieved by repeating each value in both dimensions to scale up the mask
    scale_factor = image_size // cell_size  # Calculate scale factor
    resized_mask = mask.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)

    masks = convert_idx_masks_to_bool(resized_mask[None]).to(device)
    mask_weights = (masks * attribution).sum(-1).sum(-1)
    sort_idxs = torch.argsort(mask_weights).flip(-1)
    masks = masks[sort_idxs]
    mask_weights = mask_weights[sort_idxs]
    
    topk = int(masks.shape[0] * k)
    masks_use = masks[:topk]
    mask = masks_use.sum(0)
    
    return masks, mask_weights, mask


def get_acc_frac_bright_dark(explainer_name, backbone_model, original_model, model, dark_thresh=0, bright_thresh=3, k=0.2,
                            root_dir='/shared_data0/weiqiuy/sop/exps/cosmogrid_cnn/attributions/', data_size=1000, skip=False, VERBOSE=0):
    input_dir = f'{root_dir}/{explainer_name}'
    BRIGHT_THRESHOLD = bright_thresh
    DARK_THRESHOLD = dark_thresh

    loss_all = []
    loss_original = []
    frac_bright_all = []
    frac_dark_all = []

    count = 0
    filenames = sorted(os.listdir(input_dir), key=lambda x:int(x.split('.')[0]))[:data_size]

    criterion = nn.MSELoss()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    for fi, filename in tqdm(enumerate(filenames), total=len(filenames)):
    # for i in range(len(filenames)):

        # print(filename)
        if skip and (fi // 16) % 10 != 0:
            continue
        try:
            data = torch.load(os.path.join(input_dir, filename), map_location=device)
        except:
            print('failed,', filename)
            continue
        # print(filename)
        image = data['image']
        label = data['label']
        if explainer_name == 'lime_20_test':
            lime_data = data
        if explainer_name == 'shap_20_test':
            shap_data = data
        logits = data['original_logits']
        omega_mask = data['omega_mask']
        sigma_mask = data['sigma_mask']
        omega_masks, omega_mask_weights, omega_mask_use = get_masks_from_mask(omega_mask, explainer_name)
        sigma_masks, sigma_mask_weights, sigma_mask_use = get_masks_from_mask(sigma_mask, explainer_name)

        omega_mask = omega_mask_use[None]
        sigma_mask = sigma_mask_use[None]

        unique_masks = torch.stack([omega_mask, sigma_mask])
        inputs = image
        masked_inputs = unique_masks * inputs
        outputs = backbone_model(masked_inputs)
        masked_logits = outputs.logits

        masked_logits = torch.cat([masked_logits[0,0][None], masked_logits[1,1][None]])[None]
        loss = criterion(masked_logits, label)
        loss_all.append(loss.item())
        loss_original.append(criterion(logits, label).item())

        frac_brights = []
        frac_darks = []
        for i in range(len(unique_masks)):
            k = tuple([filename, i])
            img, label, mask, mask_weight, pred = image, label, unique_masks[i], \
                        torch.tensor([1,0]) if i == 0 else torch.tensor([0,1]), logits
            # print(mask.sum(), 66*66)
            img_pth = img
            mask_pth = mask #torch.from_numpy(mask)
            masked_img = img_pth*(mask_pth > 1e-4)

            VERBOSE = False
            outputs = []

            sigma = img_pth.std()

            mask_intensity = (masked_img[(mask_pth > 1e-4)].mean())
            if VERBOSE: 
                print(f'mask #{i}')
                print(f'{frac_bright_bhuv} - fraction of bright pixels in mask / bright pixels in image')
                print(f'{frac_bright_wong} - fraction of bright pixels in mask / pixels in mask')
                print(f'Omega weight: {mask_weight[0]:.2f}, Sigma_8 weight: {mask_weight[1]:.2f}')
                print(f'[{mask_intensity:.4f} > {2*sigma:.4f}] Is supercluster? {mask_intensity > 2*sigma}')
            outputs.append([mask_intensity, 2*sigma])

            # subset for omega
            frac_bright = (masked_img > BRIGHT_THRESHOLD*sigma).sum() / ((mask_pth > 1e-4).sum())
            frac_dark = (masked_img < DARK_THRESHOLD*sigma).sum() / ((mask_pth > 1e-4).sum()) * \
                        (mask_intensity < DARK_THRESHOLD*sigma)

            frac_brights.append(frac_bright.item())
            frac_darks.append(frac_dark.item())
        frac_bright_all.append(frac_brights)
        frac_dark_all.append(frac_darks)

    print('masked loss', np.mean(loss_all), 'original loss', np.mean(loss_original))

    result = {
        'loss_all': loss_all,
        'loss_original': loss_original,
        'frac_bright_all': frac_bright_all,
        'frac_dark_all': frac_dark_all
    }
    return result


def get_cosmo_purity_results(frac_bright_all, frac_dark_all, threshold_bright=0.015, threshold_dark=0.6, verbose=0, seed=42):
    omega_bright_flat = [item[0] for item in frac_bright_all]
    omega_dark_flat = [item[0] for item in frac_dark_all]
    sigma_bright_flat = [item[1] for item in frac_bright_all]
    sigma_dark_flat = [item[1] for item in frac_dark_all]

    omega_bright_a = [item for item in omega_bright_flat if item > 0]
    omega_dark_a = [item for item in omega_dark_flat if item > 0]
    sigma_bright_a = [item for item in sigma_bright_flat if item > 0]
    sigma_dark_a = [item for item in sigma_dark_flat if item > 0]

    omega_bright = [item for item in omega_bright_a if item > threshold_bright]
    omega_dark = [item for item in omega_dark_a if item > threshold_dark]
    sigma_bright = [item for item in sigma_bright_a if item > threshold_bright]
    sigma_dark = [item for item in sigma_dark_a if item > threshold_dark]
    
    purity_mean = np.mean([
        len(omega_bright) / len(omega_bright_flat),
        len(omega_dark) / len(omega_dark_flat),
        len(sigma_bright) / len(omega_dark_flat),
        len(sigma_dark) / len(omega_dark_flat)
    ])
    
    random.seed(seed)
    means = []
    for i in range(4):
        try:
            exp_idxs = random.choices(list(range(len(omega_bright_a))), k=len(omega_bright_a))
        except:
            exp_idxs = []
        omega_bright = [omega_bright_a[ii] for ii in exp_idxs if omega_bright_a[ii] > threshold_bright]
        try:
            exp_idxs = random.choices(list(range(len(omega_dark_a))), k=len(omega_bright_a))
        except:
            exp_idxs = []
        omega_dark = [omega_dark_a[ii] for ii in exp_idxs if omega_dark_a[ii] > threshold_dark]
        try:
            exp_idxs = random.choices(list(range(len(sigma_bright_a))), k=len(omega_bright_a))
        except:
            exp_idxs = []
        sigma_bright = [sigma_bright_a[ii] for ii in exp_idxs if sigma_bright_a[ii] > threshold_bright]
        try:
            exp_idxs = random.choices(list(range(len(sigma_dark_a))), k=len(omega_bright_a))
        except:
            exp_idxs = []
        sigma_dark = [sigma_dark_a[ii] for ii in exp_idxs if sigma_dark_a[ii] > threshold_dark]
        
        mean_result = np.mean([
            len(omega_bright) / len(omega_bright_flat),
            len(omega_dark) / len(omega_dark_flat),
            len(sigma_bright) / len(omega_dark_flat),
            len(sigma_dark) / len(omega_dark_flat)
        ])
        
        means.append(mean_result)
    
    results_sum_stats_curr = {
        'omega_bright': {'count': len(omega_bright), 'mean': np.mean(omega_bright), 'std': np.std(omega_bright) },
        'omega_dark': {'count': len(omega_dark), 'mean': np.mean(omega_dark), 'std': np.std(omega_dark) },
        'sigma_bright': {'count': len(sigma_bright), 'mean': np.mean(sigma_bright), 'std': np.std(sigma_bright) },
        'sigma_dark': {'count': len(sigma_dark), 'mean': np.mean(sigma_dark), 'std': np.std(sigma_dark) },
        'purity_mean': purity_mean,
        'purity_std': np.std(means),
        'omega_bright_flat': omega_bright_flat,
        'omega_dark_flat': omega_dark_flat,
        'sigma_bright_flat': sigma_bright_flat,
        'sigma_dark_flat': sigma_dark_flat,
    }
    return results_sum_stats_curr


import random
from collections import defaultdict
import numpy as np

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from adjustText import adjust_text

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'STIXGeneral' #'Times New Roman'
})  # Set a default font size



# Function to determine Pareto frontier
def pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher Y value
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower Y value
                p_front.append(pair)
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


# threshold_bright = 0.015
# threshold_dark = 0.6

def plot_mse_vs_purity(results, threshold_bright=0.015, threshold_dark=0.6, save_dir=None, name_mapping=None):

    if name_mapping is None:
        name_mapping = {
            'lime_20_test': 'LIME-F',
            'shap_20_test': 'SHAP-F',
            'rise_20_test': 'RISE-F',
            'intgrad_20_test': 'IG-F',
            'gradcam_test': 'GC-F',
            'fullgrad_test': 'FG-F',
            'archipelago_test': 'Archi.-F',
            'mfaba_test': 'MFABA-F',
            'ampe_test': 'AMPE-F',
            'sop': 'SOP (ours)'
        }
    

    title_X = r'MSE Loss'
    title_Y = r'Purity'
    title = f'{title_X} vs {title_Y}'

    purity_means = defaultdict(list)
    loss_means = defaultdict(list)
    purity_mean = {}

    # results_baselines_all = results_all_thresholds[tuple([threshold_bright, threshold_dark])]
    # results_sum_stats = results_baselines_all

    random.seed(42)
    for i in tqdm(range(4)):
        exp_idxs = random.choices(list(range(len(results['lime_20_test']['loss_all']) - 1)), 
                                  k=len(results['lime_20_test']['loss_all']))
        for method in results:
            
            n_omega = len(results[method]['omega_bright_flat'])
            n_sigma = len(results[method]['sigma_bright_flat'])

            exp_idxs_purity_omega = random.choices(list(range(n_omega)), k=n_omega)
            exp_idxs_purity_sigma = random.choices(list(range(n_sigma)), k=n_sigma)

            omega_bright_flat = [results[method]['omega_bright_flat'][j] for j in exp_idxs_purity_omega]
            omega_dark_flat = [results[method]['omega_dark_flat'][j] for j in exp_idxs_purity_omega]
            sigma_bright_flat = [results[method]['sigma_bright_flat'][j] for j in exp_idxs_purity_sigma]
            sigma_dark_flat = [results[method]['sigma_dark_flat'][j] for j in exp_idxs_purity_sigma]

            omega_bright = [item for item in omega_bright_flat if item > threshold_bright]
            try:
                omega_dark = [item for item in omega_dark_flat if item > threshold_dark]
            except:
                import pdb; pdb.set_trace()
            sigma_bright = [item for item in sigma_bright_flat if item > threshold_bright]
            sigma_dark = [item for item in sigma_dark_flat if item > threshold_dark]

            purity = np.mean([
                len(omega_bright) / len(omega_bright_flat),
                len(omega_dark) / len(omega_dark_flat),
                len(sigma_bright) / len(omega_dark_flat),
                len(sigma_dark) / len(omega_dark_flat)
            ])

            purity_means[method].append(purity)
            # except:
            #     pass
            loss_means[method].append(np.mean([results[method]['loss_all'][j] for j in exp_idxs]))
            
            omega_bright_flat = results[method]['omega_bright_flat']
            omega_dark_flat = results[method]['omega_dark_flat']
            sigma_bright_flat = results[method]['sigma_bright_flat']
            sigma_dark_flat = results[method]['sigma_dark_flat']
            
            omega_bright_a = [item for item in omega_bright_flat if item > 0]
            omega_dark_a = [item for item in omega_dark_flat if item > 0]
            sigma_bright_a = [item for item in sigma_bright_flat if item > 0]
            sigma_dark_a = [item for item in sigma_dark_flat if item > 0]

            omega_bright = [item for item in omega_bright_a if item > threshold_bright]
            omega_dark = [item for item in omega_dark_a if item > threshold_dark]
            sigma_bright = [item for item in sigma_bright_a if item > threshold_bright]
            sigma_dark = [item for item in sigma_dark_a if item > threshold_dark]

            mean_result = np.mean([
                len(omega_bright) / len(omega_bright_flat),
                len(omega_dark) / len(omega_dark_flat),
                len(sigma_bright) / len(omega_dark_flat),
                len(sigma_dark) / len(omega_dark_flat)
            ])
            purity_mean[method] = mean_result
            
    fig, ax = plt.subplots(figsize=(2.6, 1.5))
    Xs = [np.mean(results[key]['loss_all']) for key in name_mapping]
    Ys = [purity_mean[key] for key in name_mapping]
    # Ys = [results[key]['purity_mean'] for key in name_mapping]
    Xerr = [np.std(loss_means[key]) for key in name_mapping]  # Standard deviations for X
    Yerr = [np.std(purity_means[key]) for key in name_mapping]

    labels = list(name_mapping.keys())
    pf_X, pf_Y = pareto_frontier(Xs, Ys, maxX=False, maxY=True)  # Acc is on Y-axis and we want max

    # ax.scatter(Xs[:-1], Ys[:-1], color='skyblue')
    # ax.scatter(Xs[-1:], Ys[-1:], color='r', marker='*')
    ax.errorbar(Xs[:-1], Ys[:-1], xerr=Xerr[:-1], yerr=Yerr[:-1], fmt='o', color='skyblue', markersize=2, elinewidth=1) #, ecolor='k')
    ax.errorbar(Xs[-1:], Ys[-1:], xerr=Xerr[-1:], yerr=Yerr[-1:], fmt='o', color='red', marker='*', markersize=5, elinewidth=1, ecolor='k')


    texts = []
    for i, txt in enumerate(labels):
        texts.append(ax.annotate(name_mapping[txt], (Xs[i], Ys[i])))
    adjust_text(texts) #, arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel(title.split('vs')[0].strip()) # + r' $\downarrow$')
    ax.set_ylabel(title.split('vs')[1].strip()) #+ r' $\uparrow$')
    ax.set_title(r'CosmoGrid - CNN, $\tau_v$ ' + f'{threshold_dark}, ' + r'$\tau_c$ ' + f'{threshold_bright}')
    # ax.set_ylim([0.4, 0.7])
    ax.grid(True)

    backbone_loss = np.mean(results['lime_20_test']['loss_original'])
    backbone_Xs = [backbone_loss, backbone_loss]
    backbone_Ys = [min(Ys), max(Ys)]
    ax.plot(backbone_Xs, 
            backbone_Ys, color='red', linestyle='--')
    ax.text(backbone_Xs[1], backbone_Ys[1] + 0.02, 'Backbone', color='k', ha='right', va='top')


    # Save each plot separately without directory
    
    if save_dir is not None: #cosmo_figs
        os.makedirs(save_dir, exist_ok=True)
        file_name = f'{save_dir}/{title.replace(" ", "_").lower()}_cosmogrid_tv{threshold_dark}_tc{threshold_bright}.pdf'
        print(file_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()
    return {
        'loss_mean': Xs,
        'purity_mean': Ys,
        'loss_std': Xerr,
        'purity_std': Yerr,
        'methods': [name_mapping[k] for k in name_mapping]
    }