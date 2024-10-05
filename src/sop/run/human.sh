for EXPLAINER_NAME in xdnn bagnet
do
    echo "Running $EXPLAINER_NAME"
    python /shared_data0/weiqiuy/sop/src/sop/run/gen_human_purity_egs_imagenet.py "$EXPLAINER_NAME" small
done

# lime_20 shap_20 rise_20 intgrad gradcam archipelago fullgrad mfaba agi ampe bcos xdnn bagnet sop attn