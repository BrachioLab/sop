for dirname in ./exps/sst/best/attributions/*/; do
    expln_name=$(basename $dirname)
    echo $expln_name
    python scripts/python/eval_cls_from_save.py sst $expln_name
done