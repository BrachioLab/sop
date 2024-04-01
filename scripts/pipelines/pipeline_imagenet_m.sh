# train
# python scripts/python/image/imagenet/train_multirc.py
python scripts/python/image/imagenet/train_imagenet.py --lr 0.000008 --group-gen-scale 0.02 \
--group-sel-scale 1 --group-gen-temp-alpha 1 --group-gen-temp-beta 0.1 --num-heads 1 --train-size -1 \
--val-size 1 --num-epochs 1 --scheduler-type cosine

# # save attributions sop for 5 examples each class
# python scripts/python/image/imagenet/eval_imagenet_m_sop_save.py sop 5
python scripts/python/image/imagenet/eval_imagenet_sop_save.py fresh 5


# # save attributions baselines for 5 examples each class
# methods=("lime" "archipelago" "rise" "shap" "intgrad" "gradcam")
# for method in "${methods[@]}"; do
#     python scripts/python/image/imagenet/eval_imagenet_m_baselines_save.py "$method" 5
# done

# # eval all saved attributions
# bash scripts/bash/eval_imagenet_m_from_save.sh

# # generate tables
python scripts/python/gen_results_table.py imagenet_m_2h imagenet