import json
import os
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torchmetrics.functional import f1_score, precision, recall, accuracy

results_path = Path('Results', 'mvit', 'Raw_unraveled')
results_files = os.listdir(results_path)

pred_metrics = {}
total_preds = []
total_target = []

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

        pred_metrics[filename]['macro_precision'] = precision(preds, target, average='macro', num_classes=len(classes))
        pred_metrics[filename]['micro_precision'] = precision(preds, target, average='micro', num_classes=len(classes))
        pred_metrics[filename]['macro_recall'] = recall(preds, target, average='macro', num_classes=len(classes))
        pred_metrics[filename]['micro_recall'] = recall(preds, target, average='micro', num_classes=len(classes))
        pred_metrics[filename]['macro_f1'] = f1_score(preds, target, average='macro', num_classes=len(classes))
        pred_metrics[filename]['micro_f1'] = f1_score(preds, target, average='micro', num_classes=len(classes))
        pred_metrics[filename]['accuracy'] = accuracy(preds, target)

        #import pdb; pdb.set_trace()
        print(filename, pred_metrics[filename])


    # import pdb; pdb.set_trace()
    # print(pred_results[filename])

### PLOT CONFUSION MATRIX ###
# import pdb; pdb.set_trace()
# ConfusionMatrixDisplay.from_predictions(total_target, total_preds, normalize='all')
# plt.show()

