import json
import os
from pathlib import Path
import copy
import itertools

from numpy import isin
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional import f1_score, precision, recall, accuracy
from torchmetrics.functional import mean_absolute_error, mean_squared_error

# Recursively merge two nested dicts. Dict values must be type list or dict. 
def merge_dicts(dict1, dict2):
    # Obtain all unique keys from both dicts
    keys = list(set(list(dict1.keys())+list(dict2.keys())))
    for key in keys:
        if key == 'args': continue
        if key not in dict2: continue
        elif key not in dict1: dict1[key] = dict2[key]
        else:
            # If value is list, extend. If value is dict, recurse.
            if isinstance(dict1[key], list):
                dict1[key].extend(dict2[key])
            elif isinstance(dict1[key], dict):
                dict1[key] = merge_dicts(dict1[key], dict2[key])
    return dict1
            
def main():
    results_path = Path(r'M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\Cluster_output\Results\slowfast_transfer_10_11_14\Raw')
    results_files = os.listdir(results_path)

    pred_metrics = {}
    total_preds = []
    total_target = []

    total_MAE = []
    total_acc = []

    for filename in results_files:
        pred_metrics[filename] = {}
        sub = filename.split('_')[0]
        classes = os.listdir(Path('D:\\zhaon\\Datasets\\Video Segments', sub))
        with open(Path(results_path, filename), 'r') as file:
            datalist = file.read().split('}{')
            jsonlist = []
            for i in range(len(datalist)):
                if i == 0: datalist[i] = datalist[i] + "}"
                elif i == len(datalist)-1: datalist[i] = "{" + datalist[i]
                else: datalist[i] = "{" + datalist[i] + "}"
                jsonlist.append(json.loads(datalist[i]))

            data = {}
            for dictionary in jsonlist:
                if len(dictionary['preds']) < 3: continue # Skip sanity checks
                data = merge_dicts(data, dictionary)
            # Extend nested lists into 1 list
            data['preds'] = list(itertools.chain.from_iterable(data['preds']))
            data['target'] = list(itertools.chain.from_iterable(data['target']))

            total_preds.extend(data['preds'])
            total_target.extend(data['target'])

            preds = torch.tensor(data['preds'])
            target = torch.tensor(data['target'])

            #pred_metrics[filename]['macro_precision'] = precision(preds, target, average='macro', num_classes=len(classes))
            #pred_metrics[filename]['micro_precision'] = precision(preds, target, average='micro', num_classes=len(classes))
            #pred_metrics[filename]['macro_recall'] = recall(preds, target, average='macro', num_classes=len(classes))
            #pred_metrics[filename]['micro_recall'] = recall(preds, target, average='micro', num_classes=len(classes))
            #pred_metrics[filename]['macro_f1'] = f1_score(preds, target, average='macro', num_classes=len(classes))
            #pred_metrics[filename]['micro_f1'] = f1_score(preds, target, average='micro', num_classes=len(classes))
            pred_metrics[filename]['accuracy'] = accuracy(preds, target)
            pred_metrics[filename]['MAE'] = mean_absolute_error(preds, target)
            pred_metrics[filename]['MSE'] = mean_squared_error(preds, target, squared=True)
            pred_metrics[filename]['RMSE'] = mean_squared_error(preds, target, squared=False)

            #import pdb; pdb.set_trace()
            print(filename, pred_metrics[filename])

            total_MAE.append(pred_metrics[filename]['MAE'])
            total_acc.append(pred_metrics[filename]['accuracy'])
    

    print('total acc: ', torch.mean(torch.tensor(total_acc)))
    print('total MAE: ', torch.mean(torch.tensor(total_MAE)))

        # import pdb; pdb.set_trace()
        # print(pred_results[filename])

    ### PLOT CONFUSION MATRIX ###
    # import pdb; pdb.set_trace()
    ConfusionMatrixDisplay.from_predictions(total_target, total_preds, normalize='all')
    plt.show()

if __name__ == '__main__':
    main()