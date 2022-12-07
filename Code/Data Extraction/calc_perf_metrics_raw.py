import json
import os
import re
import csv
import itertools
import pandas as pd
from pprint import pprint
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional import f1_score, precision, recall, accuracy
from torchmetrics.functional import mean_absolute_error, mean_squared_error
from filename_aggr_pred import filename_aggr_pred

# exp_name = 'mvit_08_23_14'
exp_name = 'SLOWFAST_TRANSFER_ORDINAL_SPARSE'
distr = False
# exp_name = 'slowfast_scratch_09_29_17'
if distr:   results_path = Path(r'M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\Cluster_output\Results', exp_name, 'Raw')
else:       results_path = Path('Results', exp_name, 'Raw')
results_files = os.listdir(results_path)

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

pred_metrics = {}
total_preds = []
total_target = []
top_metrics = {}
top_raws = {}
top_aggr_metrics = {}
top_aggr_raws = {}
top_aggr_errors = {}
task_data = {}
task_metrics = {}

total_MAE = []
total_acc = []
total_sub_MAE = []
total_sub_acc = []

for filename in results_files:
    pred_metrics[filename] = {}
    sub = filename.split('_')[0]
    epoch = filename.split('_')[1]
    if sub not in top_metrics: 
        top_metrics[sub] = {}
        top_raws[sub] = {}
        top_aggr_metrics[sub] = {}
    classes = os.listdir(Path('D:\\zhaon\\Datasets\\Video Segments', sub))
    with open(Path(results_path, filename), 'r') as file:
        if distr:
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
        else:
            data = json.load(file)
            # Replace printed tensor with list of ints
            for i in range(len(data['preds'])):
                if 'tensor' in data['preds'][i]:
                    pred = re.match(r"[^[]*\[([^]]*)\]", data['preds'][i]).groups()[0].split(',')
                    data['preds'][i] = [int(i) for i in pred]

        # Extend nested lists into a single list
        data['preds'] = list(itertools.chain.from_iterable(data['preds']))
        data['target'] = list(itertools.chain.from_iterable(data['target']))

        # Remove class 0 from dataset FIXME: Skip if consolidating classes
        for i in range(data['target'].count(0)):
            rm_idx = data['target'].index(0)
            data['preds'].pop(rm_idx)
            data['target'].pop(rm_idx)

        preds = torch.tensor(data['preds'])   - 1   # -1 because we are removing class 0
        target = torch.tensor(data['target']) - 1

        #pred_metrics[filename]['micro_precision'] = precision(preds, target, average='micro', num_classes=len(classes))
        #pred_metrics[filename]['micro_recall'] = recall(preds, target, average='micro', num_classes=len(classes))
        pred_metrics[filename]['macro_recall'] = recall(preds, target, average='macro', num_classes=5)
        pred_metrics[filename]['macro_precision'] = precision(preds, target, average='macro', num_classes=5)
        pred_metrics[filename]['macro_f1'] = f1_score(preds, target, average='macro', num_classes=5)
        pred_metrics[filename]['weighted_recall'] = recall(preds, target, average='weighted', num_classes=5)
        pred_metrics[filename]['weighted_precision'] = precision(preds, target, average='weighted', num_classes=5)
        pred_metrics[filename]['weighted_f1'] = f1_score(preds, target, average='weighted', num_classes=5)
        #pred_metrics[filename]['micro_f1'] = f1_score(preds, target, average='micro', num_classes=len(classes))
        pred_metrics[filename]['accuracy'] = accuracy(preds, target)
        pred_metrics[filename]['MAE'] = mean_absolute_error(preds, target)
        pred_metrics[filename]['MSE'] = mean_squared_error(preds, target, squared=True)
        pred_metrics[filename]['RMSE'] = mean_squared_error(preds, target, squared=False)

        if 'accuracy' not in top_metrics[sub] or top_metrics[sub]['MAE'] > pred_metrics[filename]['MAE'] :
            top_metrics[sub]['sub'] = int(sub.split('Sub')[1])
            top_metrics[sub]['accuracy'] = pred_metrics[filename]['accuracy'].item()
            top_metrics[sub]['MAE'] = pred_metrics[filename]['MAE'].item()
            top_metrics[sub]['macro_f1'] = pred_metrics[filename]['macro_f1'].item()
            top_metrics[sub]['macro_recall'] = pred_metrics[filename]['macro_recall'].item()
            top_metrics[sub]['macro_precision'] = pred_metrics[filename]['macro_precision'].item()
            top_metrics[sub]['weighted_f1'] = pred_metrics[filename]['weighted_f1'].item()
            top_metrics[sub]['weighted_recall'] = pred_metrics[filename]['weighted_recall'].item()
            top_metrics[sub]['weighted_precision'] = pred_metrics[filename]['weighted_precision'].item()
            top_metrics[sub]['epoch'] = epoch
            top_raws[sub]['preds'] = data['preds']
            top_raws[sub]['target'] = data['target']
            top_raws[sub]['filenames'] = data['filenames']
        
        ## Calculate AGGREGATE filename predictions ## 
        filedata = filename_aggr_pred(data['filenames'])
        aggr_preds = []
        aggr_target = []
        for f in filedata:
            aggr_preds.append(filedata[f]['pred_mode'])
            aggr_target.append(filedata[f]['target'])
        # Convert to tensor
        aggr_preds = torch.Tensor(aggr_preds).int()
        aggr_target = torch.Tensor(aggr_target).int()
        aggr_accuracy = accuracy(aggr_preds, aggr_target)
        aggr_MAE = mean_absolute_error(aggr_preds, aggr_target)
        aggr_macrof1 = f1_score(aggr_preds, aggr_target, average='macro', num_classes=6)
        aggr_macro_recall = recall(aggr_preds, aggr_target, average='macro', num_classes=6)
        aggr_macro_precision = precision(aggr_preds, aggr_target, average='macro', num_classes=6)
        aggr_weightedf1 = f1_score(aggr_preds, aggr_target, average='weighted', num_classes=6)
        aggr_weighted_recall = recall(aggr_preds, aggr_target, average='weighted', num_classes=6)
        aggr_weighted_precision = precision(aggr_preds, aggr_target, average='weighted', num_classes=6)
        if 'accuracy' not in top_aggr_metrics[sub] or top_aggr_metrics[sub]['MAE'] > aggr_MAE :
            top_aggr_metrics[sub]['sub'] = int(sub.split('Sub')[1])
            top_aggr_metrics[sub]['accuracy'] = aggr_accuracy.item()
            top_aggr_metrics[sub]['MAE'] = aggr_MAE.item()
            top_aggr_metrics[sub]['macro_f1'] = aggr_macrof1.item()
            top_aggr_metrics[sub]['macro_recall'] = aggr_macro_recall.item()
            top_aggr_metrics[sub]['macro_precision'] = aggr_macro_precision.item()
            top_aggr_metrics[sub]['weighted_f1'] = aggr_weightedf1.item()
            top_aggr_metrics[sub]['weighted_recall'] = aggr_weighted_recall.item()
            top_aggr_metrics[sub]['weighted_precision'] = aggr_weighted_precision.item()
            top_aggr_metrics[sub]['epoch'] = epoch
            top_aggr_raws[sub] = filedata
            top_aggr_errors[sub] = (aggr_target - aggr_preds).tolist()

        #import pdb; pdb.set_trace()
        # print(filename, pred_metrics[filename])

        total_MAE.append(pred_metrics[filename]['MAE'])
        total_acc.append(pred_metrics[filename]['accuracy'])

# Per-task data
for sub in top_aggr_raws:
    for file in top_aggr_raws[sub]:
        task = top_aggr_raws[sub][file]['task']
        if distr: task = '_'.join(task.split('_')[2:])      # Fix task naming discrepancy in Linux
        task = task.replace('_'+task.split('_')[-1], '')    # Remove number suffix
        if task not in task_data:
            task_data[task] = {'pred_mode':[], 'pred_mean':[], 'target':[],}
            task_metrics[task] = {}
        task_data[task]['pred_mode'].append(top_aggr_raws[sub][file]['pred_mode'])
        task_data[task]['pred_mean'].append(top_aggr_raws[sub][file]['pred_mean'])
        task_data[task]['target'].append(top_aggr_raws[sub][file]['target'])
# Calculate per-task metrics
for task in task_data:
    task_data[task]['pred_mode'] = torch.Tensor(task_data[task]['pred_mode']).int()
    task_data[task]['target'] = torch.Tensor(task_data[task]['target']).int()
    task_metrics[task]['accuracy'] = accuracy(task_data[task]['pred_mode'], task_data[task]['target']).item()
    task_metrics[task]['MAE'] = mean_absolute_error(task_data[task]['pred_mode'], task_data[task]['target']).item()
    task_metrics[task]['macro_f1'] = f1_score(task_data[task]['pred_mode'], task_data[task]['target'], average='macro', num_classes=6).item()
    task_metrics[task]['macro_recall'] = recall(task_data[task]['pred_mode'], task_data[task]['target'], average='macro', num_classes=6).item()
    task_metrics[task]['macro_precision'] = precision(task_data[task]['pred_mode'], task_data[task]['target'], average='macro', num_classes=6).item()
    task_metrics[task]['weighted_f1'] = f1_score(task_data[task]['pred_mode'], task_data[task]['target'], average='weighted', num_classes=6).item()
    task_metrics[task]['weighted_recall'] = recall(task_data[task]['pred_mode'], task_data[task]['target'], average='weighted', num_classes=6).item()
    task_metrics[task]['weighted_precision'] = precision(task_data[task]['pred_mode'], task_data[task]['target'], average='weighted', num_classes=6).item()
    
# pprint(top_metrics)
print('total acc: ', torch.mean(torch.tensor(total_acc)))
print('total MAE: ', torch.mean(torch.tensor(total_MAE)))

## Save metrics to csv ##
metric_table = pd.DataFrame(top_metrics).T.sort_values(by='sub')
avg_row = {'sub':'avg',
           'accuracy':metric_table['accuracy'].mean(),
           'MAE':metric_table['MAE'].mean(),
           'macro_f1':metric_table['macro_f1'].mean(),
           'macro_recall':metric_table['macro_recall'].mean(),
           'macro_precision':metric_table['macro_precision'].mean(),
           'weighted_f1':metric_table['weighted_f1'].mean(),
           'weighted_recall':metric_table['weighted_recall'].mean(),
           'weighted_precision':metric_table['weighted_precision'].mean(),
           'epoch':None}
metric_table = pd.concat([metric_table, pd.DataFrame(avg_row,index=['avg'])]) # Append avg row
savepath = Path(results_path, '..', 'Extracted')
os.makedirs(savepath, exist_ok=True)
savefile = Path(savepath, f'{exp_name}_TopMetrics.csv')
metric_table.to_csv(savefile)

## Save AGGREGATE metrics to csv
aggr_metric_table = pd.DataFrame(top_aggr_metrics).T.sort_values(by='sub')
aggr_avg_row = {'sub':'avg',
                'accuracy':aggr_metric_table['accuracy'].mean(),
                'MAE':aggr_metric_table['MAE'].mean(),
                'macro_f1':aggr_metric_table['macro_f1'].mean(),
                'macro_recall':aggr_metric_table['macro_recall'].mean(),
                'macro_precision':aggr_metric_table['macro_precision'].mean(),
                'weighted_f1':aggr_metric_table['weighted_f1'].mean(),
                'weighted_recall':aggr_metric_table['weighted_recall'].mean(),
                'weighted_precision':aggr_metric_table['weighted_precision'].mean(),
                'epoch':None}
aggr_metric_table = pd.concat([aggr_metric_table, pd.DataFrame(aggr_avg_row,index=['avg'])]) # Append avg row
aggr_savefile = Path(savepath, f'{exp_name}_AggrTopMetrics.csv')
aggr_metric_table.to_csv(aggr_savefile)

## Save Per-TASK metrics to csv
task_metric_table = pd.DataFrame(task_metrics).T
task_avg_row = {'accuracy':task_metric_table['accuracy'].mean(),
                'MAE':task_metric_table['MAE'].mean(),
                'macro_f1':task_metric_table['macro_f1'].mean(),
                'macro_recall':task_metric_table['macro_recall'].mean(),
                'macro_precision':task_metric_table['macro_precision'].mean(),
                'weighted_f1':task_metric_table['weighted_f1'].mean(),
                'weighted_recall':task_metric_table['weighted_recall'].mean(),
                'weighted_precision':task_metric_table['weighted_precision'].mean(),
                }
task_metric_table = pd.concat([task_metric_table, pd.DataFrame(task_avg_row, index=['avg'])])
task_savefile = Path(savepath, f'{exp_name}_TaskAggrMetrics.csv')
task_metric_table.to_csv(task_savefile)

## Save top aggr errors to csv
error_table = pd.DataFrame.from_dict(top_aggr_errors, orient='index').T
error_total_row = pd.DataFrame(error_table.stack().values, columns=['total'])
error_table = pd.concat([error_table, error_total_row])
error_savefile = Path(savepath, f'{exp_name}_TopAggrErrors.csv')
error_table.to_csv(error_savefile)

    # import pdb; pdb.set_trace()
    # print(pred_results[filename])

### PLOT CONFUSION MATRIX ###
# for sub in top_raws:
#     total_preds.extend(top_raws[sub]['preds'])
#     total_target.extend(top_raws[sub]['target'])
# ConfusionMatrixDisplay.from_predictions(total_target, total_preds, normalize='all')

## PLOT DATA TABLE ### 
# rounded_table = aggr_metric_table.round(decimals=3)
# fig, ax = plt.subplots()
# fig.patch.set_visible(False)
# ax.axis('off')
# table = plt.table(cellText=rounded_table.values, colLabels=rounded_table.columns, loc="center")
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1,1.2)

# plt.show()

