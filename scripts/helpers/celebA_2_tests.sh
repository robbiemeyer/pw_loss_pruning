#!/bin/bash

method=$1
run_name=$2

model="celeba_2"
gpu_id=${GPU_ID:-0}

name="${method}.csv"
dir="output/results/$model/$run_name"
output_dir="output/pruned_models/$model/$run_name"

props="0.9375 0.9688 0.9844 0.9922 0.9961"

batch_size=64

# Autobot parameters
ab_lr=1.81
ab_beta=3.07
ab_gamma=0 #0.18
ab_iters=250

# Taylor Parameters
tay_lr=0.01
tay_batches=4
tay_ckpt_name=$(date +%Y-%m-%d_%H:%M)

# Score Weighted Params
sw_min_weight=0.75
sw_gamma=3
if [ $method = "sw_taylor" ]
then
  sw_min_weight=0.9
  sw_gamma=5
fi

function test_props () {
  output_name=$1
  comment=$2
  for prop in $props
  do
      echo Running $comment - $prop "(${output_name})"
      output="${output_dir}/${method}_$prop.pt"
      python -m scripts.prune --model $model --output $output_name --target_reduction $prop \
        --comment "$comment" --method $method --sw_gamma $sw_gamma --sw_min_weight $sw_min_weight \
        --ab_beta $ab_beta --ab_iters $ab_iters --ab_gamma $ab_gamma --ab_lr $ab_lr \
        --tay_lr $tay_lr --tay_batches $tay_batches --tay_ckpt_name $tay_ckpt_name \
        --gpu_id $gpu_id --batch_size $batch_size --pruned_model_output $output #--validation
      echo -e "\n"
  done
}

mkdir -p $dir
mkdir -p $output_dir
test_props "$dir/$name" " "
