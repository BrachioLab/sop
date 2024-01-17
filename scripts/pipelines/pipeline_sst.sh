# train
# python scripts/python/train_multirc.py

# save attributions sop for 5 examples each class
python scripts/python/text/multirc/eval_multirc_sop_save.py sop

# save attributions baselines for 5 examples each class
# methods=("lime" "archipelago" "rise" "shap" "intgrad" "gradcam")
methods=("lime" "rise" "shap" "intgrad")
for method in "${methods[@]}"; do
    python scripts/python/text/multirc/eval_sst_baselines_save.py "$method"
done

# eval all saved attributions
bash scripts/bash/eval_sst_from_save.sh

# generate tables
python scripts/python/gen_results_table.py sst sst