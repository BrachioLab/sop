# train
# python scripts/python/train_multirc.py
python train_sst.py --group-gen-scale 2 --group-sel-scale 1 --num-heads 1 \
--group-gen-temp-alpha 1 --group-gen-temp-beta 0.1

# save attributions sop for 5 examples each class
python scripts/python/text/sst/eval_sst_sop_save.py sop

# save attributions baselines for 5 examples each class
# methods=("lime" "archipelago" "rise" "shap" "intgrad" "gradcam")
methods=("lime" "rise" "shap" "intgrad")
for method in "${methods[@]}"; do
    python scripts/python/text/sst/eval_sst_baselines_save.py "$method"
done

# eval all saved attributions
bash scripts/bash/eval_sst_from_save.sh

# generate tables
python scripts/python/gen_results_table.py sst sst