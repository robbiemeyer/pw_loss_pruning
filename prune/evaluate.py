from sklearn.metrics import roc_auc_score
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from time import time

import torch

from prune.utils import get_all_predictions

def evaluate(outputs, labels, model):
    metrics = {}
    metrics['auroc'] = roc_auc_score(labels, model.reduce_fn(outputs), multi_class='ovr')
    metrics['accuracy'] = (model.pred_fn(outputs) == labels).float().mean().item()

    classwise_auroc = roc_auc_score(labels, model.reduce_fn(outputs), multi_class='ovr',
                                    average=None)
    for i, v in enumerate(classwise_auroc):
        metrics['{}_auroc'.format(i)] = v

    #for target in labels.unique():
    #    included = labels == target
    #    predicted = model.pred_fn(outputs) == target
    #    metrics[str(target.item()) + '_error'] = \
    #        1 - ((included & predicted).float().sum() / included.float().sum()).item()
    return metrics

def evaluate_model(model, test_datas, target_reduction, input_shape, eval_on_all=False, comment='',
        batch_size=256, num_workers=6, report_flops=True, show_flop_warnings=False):

    num_params = sum([param.numel() for param in model.parameters()])

    if report_flops:
        inp = torch.rand(1, *input_shape, device=model.device)
        flops = FlopCountAnalysis(model, inp)

        flops.uncalled_modules_warnings(show_flop_warnings)
        flops.unsupported_ops_warnings(show_flop_warnings)

    common_results = {
        'target_reduction': round(target_reduction, 4),
        'num_params': num_params,
        'flops': flops.total() if report_flops else None,
        'comment': comment
    }

    results = []
    if eval_on_all:
        all_outputs, all_labels = [], []
        total_duration = 0

    for group, test_data in test_datas.items():
        labels, outputs = get_all_predictions(test_data, model, cache=False, raw=True,
                                              batch_size=batch_size, num_workers=num_workers)

        if eval_on_all:
            all_outputs.append(outputs)
            all_labels.append(labels)

        results.append(
            {'group': group} | common_results | evaluate(outputs, labels, model)
        )

    if eval_on_all:
        results.append(
            {'group': 'all', 'duration': total_duration} | common_results | \
                evaluate(torch.cat(all_outputs), torch.cat(all_labels), model)
        )

    return pd.DataFrame(results)
