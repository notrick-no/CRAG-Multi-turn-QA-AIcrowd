#!/bin/bash
export TF_ENABLE_ONEDNN_OPTS=0

project_dir=$(dirname "$(realpath "$0")")
cd $project_dir
echo "start testing"
echo "................"

python local_evaluation.py \
    --dataset-type single-turn \
    --split validation \
    --num-conversations 100 \
    --display-conversations 3 \
    --eval-model gpt-4o-mini \
    --suppress-web-search-api # Only include when evaluatingn for Single Source Augmentation Track, where web-search-api is not available but image-search-api is available

echo "................"
echo "end testing"

chmod +x run_evaluation.sh 