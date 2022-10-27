import math
def filename_aggr_pred(filenames):
    for file in filenames:
        sublen = len(file.split('\\')[-1].split('_')[0])
        task = file.split('\\')[-1][sublen+1:]
        filenames[file]['task'] = task

        if isinstance(filenames[file]['preds'][0], str):
            if 'tensor' in filenames[file]['preds'][0]:
                # Convert list of strings in tensor format to list of ints
                l = []
                for pred in (filenames[file]['preds']):
                    l.append(int(pred.split('(')[1][0]))
                filenames[file]['preds'] = l
        
        # Mode
        filenames[file]['pred_mode'] = max(set(filenames[file]['preds']), key=filenames[file]['preds'].count)
        # Rounded Mean
        filenames[file]['pred_mean'] = round(sum(filenames[file]['preds'])/len(filenames[file]['preds']))

        # Reduce target to single int
        filenames[file]['target'] = filenames[file]['target'][0]
    return filenames
                

