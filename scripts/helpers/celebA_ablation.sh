#!/bin/bash

run_number=$1
model="celeba_1"
gpu_id=${GPU_ID:-0}

dir="output/results/$model/ablations_$run_number"
output_dir="output/pruned_models/$model/ablations_$run_number"

props="0.9375 0.9688 0.9844 0.9922 0.9961"

batch_size=64

# Autobot parameters
ab_lr=0.85
ab_beta=2.7
ab_gamma=0 #0.1
ab_iters=200

# Score Weighted Params
sw_min_weight=0.3 #0.75 #0.7 #0.6
sw_gamma=1

function test_props () {
  output_name=$1
  method=$2
  use_soft_retrain=$3
  fix_labels=$4
  for prop in $props
  do
      echo Running $prop "(${output_name})"
      output="${output_dir}/${method}_$prop.pt"
      python -m scripts.prune --model $model --output $output_name --target_reduction $prop \
        --method $method --sw_gamma $sw_gamma --sw_min_weight $sw_min_weight \
        --ab_beta $ab_beta --ab_iters $ab_iters --ab_gamma $ab_gamma --ab_lr $ab_lr \
        --gpu_id $gpu_id --batch_size $batch_size --pruned_model_output $output \
        --use_soft_retrain $use_soft_retrain --fix_labels $fix_labels
      echo -e "\n"
  done
}

mkdir -p $dir
mkdir -p $output_dir

#test_props "$dir/weight_only_no_retrain.csv" "sw_autobot" 0 0
#test_props "$dir/corr_only_no_retrain.csv" "autobot" 0 1
#test_props "$dir/weight_only_retrain.csv" "sw_autobot" 1 0
#test_props "$dir/corr_only_retrain.csv" "autobot" 1 1

test_props "$dir/taylor_weight_only_retrain.csv" "sw_taylor" 1 0
test_props "$dir/taylor_corr_only_retrain.csv" "taylor" 1 1
