from decimal import DivisionByZero
import json
import os
from pathlib import Path


results_path = Path('Results', 'slowfast', 'Raw')
results_files = os.listdir(results_path)

out_path = Path('Results', 'slowfast', 'Raw_unraveled_renamed')


for filename in results_files:
    with open(Path(results_path, filename)) as file:
        data = json.load(file)
        # import pdb; pdb.set_trace()

        sub = filename.split('_')[0]
        classes = os.listdir(Path('Video Segments', sub))

        #flatten preds and target lists
        flat_preds = [x for xs in data['preds'] for x in xs]
        flat_target = [x for xs in data['target'] for x in xs]

        for i in range(len(flat_preds)):
            flat_preds[i] = int(classes[flat_preds[i]])
            flat_target[i] = int(classes[flat_target[i]])

        data['preds'] = flat_preds
        data['target'] = flat_target

        #save
        os.makedirs(out_path, exist_ok=True)
        savefile = Path(out_path, filename)
        with open(savefile, 'w') as f:
            json.dump(data, f, indent=4)
        print('Saved to ', savefile)

