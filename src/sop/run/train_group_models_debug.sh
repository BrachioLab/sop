for EXPLAINER_NAME in intgrad gradcam xdnn bcos ampe agi attn fullgrad mfaba sop
do
    echo "Running $EXPLAINER_NAME"
    python /shared_data0/weiqiuy/sop/src/sop/run/train_group_models.py --method "$EXPLAINER_NAME" --lr 0.001 --num_epochs 1 --debug
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