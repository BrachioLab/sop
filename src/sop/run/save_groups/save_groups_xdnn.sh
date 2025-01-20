for EXPLAINER_NAME in xdnn #bagnet sop
do
    echo "Running $EXPLAINER_NAME"
    python /shared_data0/weiqiuy/sop/src/sop/run/save_groups.py --method "$EXPLAINER_NAME" #--skip_saved
done

# methods = [
#     'bcos',
#     'xdnn',
#     'bagnet',
#     'sop',
#     'shap',
#     'rise',
#     'lime',
#     'fullgrad',
#     'gradcam',
#     'intgrad',
#     'attn',
#     'archipelago',
#     'mfaba',
#     'agi',
#     'ampe',
# ]