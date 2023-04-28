#!/bin/bash

method=$1
run_name=$2

model="cifar10_1_shortcut"
gpu_id=${GPU_ID:-0}

name="${method}.csv"
dir="output/results/$model/$run_name"
output_dir="output/pruned_models/$model/$run_name"

props="0.25 0.5 0.75 0.875 0.9375"

batch_size=128

# Score Weighted Params
sw_min_weight=0.5
sw_gamma=1

# Autobot parameters
ab_lr=0.7
ab_iters=200
ab_beta=5.3
ab_gamma=0

# Taylor Parameters
tay_lr=0.01
tay_batches=4
tay_ckpt_name=$(date +%Y-%m-%d_%H:%M)

function test_props () {
  output_name=$1
  comment="$comment $2"
  for prop in $props
  do
      echo Running $comment - $prop "(${output_name})"
      output="${output_dir}/${method}_$prop.pt"
      python -m scripts.prune --model $model --output $output_name --target_reduction $prop \
        --comment "$comment" --method $method --sw_gamma $sw_gamma --sw_min_weight $sw_min_weight \
        --ab_beta $ab_beta --ab_iters $ab_iters --ab_gamma $ab_gamma --ab_lr $ab_lr \
        --tay_lr $tay_lr --tay_batches $tay_batches --tay_ckpt_name $tay_ckpt_name \
        --gpu_id $gpu_id --batch_size $batch_size --pruned_model_output $output #--validation
  done
}

mkdir -p $dir
mkdir -p $output_dir
test_props "$dir/$name" " "
