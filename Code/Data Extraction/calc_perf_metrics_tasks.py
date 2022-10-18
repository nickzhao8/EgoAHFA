from decimal import DivisionByZero
import json
import os
from pathlib import Path
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


results_path = Path('Results', 'slowfast', 'Raw')
results_files = os.listdir(results_path)

pred_results = {}

for filename in results_files:
    # pred_results[filename] = {}
    sub = filename.split('_')[0]
    classes = os.listdir(Path('Video Segments', sub))
    with open(Path(results_path, filename)) as file:
        data = json.load(file)

        # import pdb; pdb.set_trace()
        for task in data['tasks'].keys():
            task_ind = task
            # Replace incorrectly-named tasks
            if task == 'Dice':
                task_ind = 'Dice roll'
            elif task == 'Reading' or task == 'Book':
                task_ind = 'Read book'
            elif task == 'Eat chips':
                task_ind = 'Chips'
            elif task == 'Tablet':
                continue
            elif task == 'Pen' or task == 'Pen ':
                task_ind = 'Pencil'
            elif task == 'Drink water':
                task_ind = 'Drink'

            if task_ind not in pred_results.keys():
                pred_results[task_ind] = {'preds':[], 'target':[]}
                print(filename, task_ind)

            # Convert class indices to actual class labels. 
            for i in range(len(data['tasks'][task]['preds'])):
                try: data['tasks'][task]['preds'][i] = int(classes[data['tasks'][task]['preds'][i]])
                except IndexError: import pdb; pdb.set_trace()
                data['tasks'][task]['target'][i] = int(classes[data['tasks'][task]['target'][i]])

            pred_results[task_ind]['preds'].extend(data['tasks'][task]['preds'])
            pred_results[task_ind]['target'].extend(data['tasks'][task]['target'])


pred_metrics = {}
for task in pred_results:
    pred_metrics[task] = {}
    #import pdb; pdb.set_trace()
    preds = torch.tensor(pred_results[task]['preds'])
    target = torch.tensor(pred_results[task]['target'])

    macro_prf = precision_recall_fscore_support(preds, target, average='macro')
    micro_prf = precision_recall_fscore_support(preds, target, average='micro')

    pred_metrics[task]['macro_precision'] = macro_prf[0]
    pred_metrics[task]['macro_recall'] = macro_prf[1]
    pred_metrics[task]['macro_f1'] = macro_prf[2]
    pred_metrics[task]['micro_precision'] = micro_prf[0]
    pred_metrics[task]['micro_recall'] = micro_prf[1]
    pred_metrics[task]['micro_f1'] = micro_prf[2]
    pred_metrics[task]['accuracy'] = accuracy_score(target,preds)

    #import pdb; pdb.set_trace()

savefile = Path('Results', 'slowfast', 'task_metrics.json')
with open(savefile, 'w') as f:
    json.dump(pred_metrics, f, indent=4)


