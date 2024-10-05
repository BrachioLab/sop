# test for attributions and group attributions
# move later

from tqdm.auto import tqdm
import time

def test_explainer(dataloader, explainer_name, config, strict=True):
    explainer = get_explainer(explainer_name)
    num_classes = config['model']['num_classes']
    
    # for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    batch = next(iter(dataloader))
    inputs, labels = batch
    inputs, labels = inputs.to(device), labels.to(device)
    bsz = inputs.shape[0]

    with torch.no_grad():
        logits = original_model(inputs)
        preds = torch.argmax(logits, dim=-1)
        probs = logits.softmax(-1)

        aggr_preds = []
        explns = []
        multi_labels = torch.tensor([list(range(num_classes))]).to(device)
    
        j = 0

        start = time.time()
        # test attributions
        expln = explainer(inputs[j:j+1].clone(), multi_labels.clone(), return_groups=False)
        print('attributions', time.time() - start)
        print(expln.attributions.shape)

        # test group attributions
        start = time.time()
        expln = explainer(inputs[j:j+1].clone(), multi_labels.clone(), return_groups=True)
        print('group attributions', time.time() - start)
        print(expln.attributions.shape, expln.group_masks.shape, expln.group_attributions.shape)
