import json
import argparse
import os
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs="*")
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    print(args)
    dirs = args.dirs
    output = args.output

    with open(output, 'w') as f_out: 
        complete_metrics = defaultdict(list)
        for dir in dirs:
            dir_metrics = {}
            results_files = [f for f in os.listdir(dir) if f.endswith('results.json')]
            for f in results_files:
                f_metrics = open(os.path.join(dir, f))
                metrics_dict = json.load(f_metrics)
                metrics = metrics_dict['metrics']
                dir_metrics[f] = {
                    'roc_auc': metrics['bootstrap_roc_auc_mean'],
                    'std': metrics['bootstrap_roc_auc_std'],
                    'tpr_at_low_fpr': metrics['tpr_at_low_fpr']
                }


            target = os.path.split(os.path.split(dir)[0])[-1]
            dataset = os.path.split(dir)[-1]
            dataset = dataset.split('-')[-1]
            dir_metrics['target'] = target
            complete_metrics[dataset].append(dir_metrics)
        f_out.write(json.dumps(complete_metrics, indent=4))
