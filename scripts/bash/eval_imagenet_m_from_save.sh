for dirname in ./exps/imagenet_m_2h/best/attributions/*/; do
    expln_name=$(basename $dirname)
    echo $expln_name
    python scripts/python/image/imagenet/eval_imagenet_m_from_save_all_metrics.py $expln_name
done