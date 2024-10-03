for EXPLAINER_NAME in agi #ampe # lime shap rise intgrad gradcam archipelago fullgrad
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
#     # 'agi', 
#     # 'ampe',
#     # 'bcos',
#     # 'xdnn',
#     # 'bagnet'
# ]