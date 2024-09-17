for EXPLAINER_NAME in fullgrad mfaba agi ampe bcos xdnn bagnet; do
    echo "Running $EXPLAINER_NAME"
    python /shared_data0/weiqiuy/sop/src/sop/run/fidelity.py "$EXPLAINER_NAME"
done

# explainer_names = [
#     'lime',
#     'shap',
#     'rise',
#     'intgrad',
#     'gradcam',
#     'archipelago',
#     'fullgrad',
#     # 'attn', # need to make it an actual model
#     'mfaba',
#     'agi',
#     'ampe',
#     'bcos',
#     'xdnn',
#     'bagnet'
# ]