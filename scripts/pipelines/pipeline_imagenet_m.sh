# train
python scripts/python/image/imagenet/train_multirc.py

# # save attributions sop for 5 examples each class
# python scripts/python/image/imagenet/eval_imagenet_m_sop_save.py sop 5

# # save attributions baselines for 5 examples each class
# methods=("lime" "archipelago" "rise" "shap" "intgrad" "gradcam")
# for method in "${methods[@]}"; do
#     python scripts/python/image/imagenet/eval_imagenet_m_baselines_save.py "$method" 5
# done

# # eval all saved attributions
# bash scripts/bash/eval_imagenet_m_from_save.sh

# # generate tables
python scripts/python/gen_results_table.py imagenet_m_2h imagenet