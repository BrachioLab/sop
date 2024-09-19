for EXPLAINER_NAME in ampe # lime shap rise intgrad gradcam archipelago fullgrad
do
    echo "Running $EXPLAINER_NAME"
    python /shared_data0/weiqiuy/sop/src/sop/run/cosmogrid_save.py "$EXPLAINER_NAME"
done

# still need to add sop

# explainer_names = [
#     'lime',
#     'shap',
#     'rise',
#     'intgrad',
#     'gradcam',
#     'archipelago',
#     'fullgrad',
#     # 'attn', # don't work with non transformer
#     'mfaba',
#     # 'agi', # only have implementation for classification
#     # 'ampe',
#     # 'bcos',
#     # 'xdnn',
#     # 'bagnet'
# ]