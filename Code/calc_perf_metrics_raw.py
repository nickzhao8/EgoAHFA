import json
import os
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional import f1_score, precision, recall, accuracy
from torchmetrics.functional import mean_absolute_error, mean_squared_error

results_path = Path('Results', 'mvit_08_23_14', 'Raw_unraveled')
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
    with open(Path(results_path, filename)) as file:
        data = json.load(file)
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

