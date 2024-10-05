for dirname in ./exps/multirc_5e-6/best/attributions/*/; do
    expln_name=$(basename $dirname)
    echo $expln_name
    python scripts/python/eval_cls_from_save.py multirc_5e-06 $expln_name
done