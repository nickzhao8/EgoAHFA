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
exp_name = 'slowfast_scratch_11_07_17'
distr = True
# exp_name = 'slowfast_scratch_09_29_17'
# results_path = Path('Results', exp_name, 'Raw')
results_path = Path(r'M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\Cluster_output\Results', exp_name, 'Raw')
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
        top_aggr_raws[sub] = {}
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

        preds = torch.tensor(data['preds'])
        target = torch.tensor(data['target'])

        #pred_metrics[filename]['macro_precision'] = precision(preds, target, average='macro', num_classes=len(classes))
        #pred_metrics[filename]['micro_precision'] = precision(preds, target, average='micro', num_classes=len(classes))
        #pred_metrics[filename]['macro_recall'] = recall(preds, target, average='macro', num_classes=len(classes))
        #pred_metrics[filename]['micro_recall'] = recall(preds, target, average='micro', num_classes=len(classes))
        pred_metrics[filename]['macro_f1'] = f1_score(preds, target, average='macro', num_classes=6)
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
        if 'accuracy' not in top_aggr_metrics[sub] or top_aggr_metrics[sub]['MAE'] > aggr_MAE :
            top_aggr_metrics[sub]['sub'] = int(sub.split('Sub')[1])
            top_aggr_metrics[sub]['accuracy'] = aggr_accuracy.item()
            top_aggr_metrics[sub]['MAE'] = aggr_MAE.item()
            top_aggr_metrics[sub]['macro_f1'] = aggr_macrof1.item()
            top_aggr_metrics[sub]['epoch'] = epoch
            top_aggr_raws['preds'] = data['preds']
            top_aggr_raws['target'] = data['target']
            top_aggr_raws['filenames'] = filedata

        #import pdb; pdb.set_trace()
        # print(filename, pred_metrics[filename])

        total_MAE.append(pred_metrics[filename]['MAE'])
        total_acc.append(pred_metrics[filename]['accuracy'])
    
# pprint(top_metrics)
print('total acc: ', torch.mean(torch.tensor(total_acc)))
print('total MAE: ', torch.mean(torch.tensor(total_MAE)))

## Save metrics to csv ##
metric_table = pd.DataFrame(top_metrics).T.sort_values(by='sub')
avg_row = {'sub':'avg',
           'accuracy':metric_table['accuracy'].mean(),
           'MAE':metric_table['MAE'].mean(),
           'macro_f1':metric_table['macro_f1'].mean(),
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
                'epoch':None}
aggr_metric_table = pd.concat([aggr_metric_table, pd.DataFrame(aggr_avg_row,index=['avg'])]) # Append avg row
aggr_savefile = Path(savepath, f'{exp_name}_AggrTopMetrics.csv')
aggr_metric_table.to_csv(aggr_savefile)


    # import pdb; pdb.set_trace()
    # print(pred_results[filename])

### PLOT CONFUSION MATRIX ###
# import pdb; pdb.set_trace()
# for sub in top_raws:
#     total_preds.extend(top_raws[sub]['preds'])
#     total_target.extend(top_raws[sub]['target'])
# ConfusionMatrixDisplay.from_predictions(total_target, total_preds, normalize='all')

## PLOT DATA TABLE ### 
rounded_table = aggr_metric_table.round(decimals=3)
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
# ax.axis('tight')
table = plt.table(cellText=rounded_table.values, colLabels=rounded_table.columns, loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1,1.2)
# fig.tight_layout()

plt.show()

