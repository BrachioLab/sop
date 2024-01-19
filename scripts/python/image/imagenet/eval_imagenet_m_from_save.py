import json
import torch
import os
import sys
from tqdm.auto import tqdm
import json

sys.path.append('lib/exlib/src')


if __name__ == '__main__':
    explainer_name = sys.argv[1]
    attrs_dir = 'exps/imagenet_m_2h/best/attributions/{}'.format(explainer_name)
    results_dir = 'exps/imagenet_m_2h/best/results_'

    filenames_sorted = sorted(os.listdir(attrs_dir), key=lambda x:int(x.split('.')[0]))

    kl_divs = []
    fidelities = []
    accuracies = []
    for filename in tqdm(filenames_sorted):
        attributions_results = torch.load(os.path.join(attrs_dir, filename))

        logit = attributions_results['logit']
        label = attributions_results['label']
        grouped_attrs_aggr = attributions_results['grouped_attrs_aggr']

        prob = torch.softmax(logit, dim=-1)
        aggr_prob = torch.softmax(grouped_attrs_aggr, dim=-1)

        pred = torch.argmax(prob).item()
        aggr_pred = torch.argmax(aggr_prob).item()

        # compute KL divergence for prob and aggr_prob
        kl_div = torch.nn.functional.kl_div(torch.log(prob), aggr_prob, reduction='sum')
        kl_divs.append(kl_div.item())

        # compute consistency for pred and aggr_pred
        fidelity = (pred == aggr_pred)
        fidelities.append(fidelity)

        # compute accuracy for pred and label
        accuracy = (pred == label).item()
        accuracies.append(accuracy)

        import pdb; pdb.set_trace()
    
    kl_div_mean = sum(kl_divs) / len(kl_divs)
    fidelity_mean = sum(fidelities) / len(fidelities)
    accuracy_mean = sum(accuracies) / len(accuracies)
    print('explainer_name: {}'.format(explainer_name))
    print('soft_fidelity_loss_mean: {}'.format(kl_div_mean))
    print('fidelity_mean: {}'.format(fidelity_mean))
    print('accuracy_mean: {}'.format(accuracy_mean))

    results = {
        'soft_fidelity_loss_mean': kl_div_mean,
        'fidelity_mean': fidelity_mean,
        'accuracy_mean': accuracy_mean
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, explainer_name + '_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
