for EXPLAINER_NAME in bagnet xdnn bcos; do
    echo "Running $EXPLAINER_NAME"
    python /shared_data0/weiqiuy/sop/src/sop/run/fidelity.py "$EXPLAINER_NAME"
done