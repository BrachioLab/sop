for EXPLAINER_NAME in rise; do
    echo "Running $EXPLAINER_NAME"
    python /shared_data0/weiqiuy/sop/src/sop/run/fidelity_multirc.py "$EXPLAINER_NAME"
done

# still need to do fresh and sop

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