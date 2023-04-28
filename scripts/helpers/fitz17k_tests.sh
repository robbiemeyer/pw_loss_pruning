#!/bin/bash

props="0.15 0.45 0.75"
props=0.15
model="CNN/checkpoints/last.ckpt"

alpha=1
beta=10

function test_props () {
  output_name=$1
  alpha=$2
  beta=$3
  comment=$4
  for prop in $props
  do
      echo Running $prop "(${output_name})"
      python -m CNN.prune.prune_fitz --model $model --output $output_name \
        --target_reduction $prop --alpha $alpha --beta $beta --comment "$comment" --no_weights
      echo -e "\n"
  done
}

test_props "outputs/test.csv" $alpha $beta " "
#test_props "outputs/fitz_tests_nw.csv" $alpha $beta " "
#test_props "outputs/fitz_tests_baseline.csv" $alpha $beta " "
