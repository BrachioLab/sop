for dirname in ./exps/imagenet_m_2h/best/attributions/*/; do
    expln_name=$(basename $dirname)
    echo $expln_name
    python scripts/python/eval_cls_from_save.py imagenet_m_2h $expln_name
done