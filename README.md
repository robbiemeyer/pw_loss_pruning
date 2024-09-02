# Code for "A Fair Loss Function for Network Pruning"

This repository contains the code for the paper "A Fair Loss Function for Network Pruning."

`scripts/train.py` is a basic Pytorch Lightning script that can be customized to train unpruned
models.

`scripts/prune.py` can be used to prune and evaluate saved models. It supports the datasets and
models referenced in the paper. Important parameters, such as filepaths, should be set in
`params.py` before running the script.

The script can be run using `python -m scripts.prune` and the following options are available:

```
  -h, --help            show help message and exit
  --model {celeba_1,celeba_2,cifar10_1,cifar10_2,fitz17k_1,fitz17k_2,celeba_1_alt1,celeba_1_alt2,celeba_1_alt3,cifar10_1_imbal,cifar10_1_shortcut}
  --gpu_id GPU_ID       The id of the GPU to train on.
  --output OUTPUT       The filename to output results to
  --target_reduction TARGET_REDUCTION
                        The proportion of parameters to remove
  --pruned_model_output PRUNED_MODEL_OUTPUT
  --method {magnitude,random,autobot,sw_autobot,taylor,sw_taylor}
  --validation
  --sw_gamma SW_GAMMA
  --sw_min_weight SW_MIN_WEIGHT
  --ab_beta AB_BETA
  --ab_gamma AB_GAMMA
  --ab_iters AB_ITERS
  --ab_lr AB_LR
  --tay_batches TAY_BATCHES
  --tay_prunes_per_iter TAY_PRUNES_PER_ITER
  --tay_lr TAY_LR
  --tay_ckpt_name TAY_CKPT_NAME
  --batch_size BATCH_SIZE
  --n_prunes N_PRUNES
  --epochs_per_prune EPOCHS_PER_PRUNE
  --comment COMMENT
  --fix_labels {-1,0,1}
  --log_name LOG_NAME
  --use_soft_retrain {-1,0,1}
  --use_weights_retrain {-1,0,1}
```

## Dependencies

- pytorch 1.12.1
- torchvision 0.13.1
- pytorch-lightning 1.7.1
- torch-pruning 1.0.0
- pillow
- fvcore
- scikit-learn
