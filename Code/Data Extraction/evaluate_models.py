from unittest import result
import torch
import flash
from flash.video import VideoClassificationData, VideoClassifier
import fiftyone as fo
from pathlib import Path
import os
from datetime import datetime
import json

def main():
    dataset_root = Path("Uniform Video Samples", "30_frames_10_stride")
    modelarch = 'slow_r50_fit'
    models_root = Path('Models', modelarch)
    subdirs = os.listdir(dataset_root)
    models = os.listdir(models_root)
    datasets = []
    model_evaluations = {}
    classes = ['1','2','3','4','5']

    # 1. Load and partition dataset
    for subdir in subdirs:
        #dataset = fo.Dataset.from_dir(
        #    dataset_dir=Path(dataset_root, subdir),
        #    dataset_type=fo.types.VideoClassificationDirectoryTree,
        #    name="uniform_"+subdir,
        #)
        #dataset.persistent=False
        dataset = fo.load_dataset("uniform_"+subdir)
        dataset.persistent=True
        datasets.append(dataset)

    for subdir in subdirs:
        val_sub = subdir
        train_dataset = fo.Dataset("train_dataset_"+str(val_sub), overwrite=True)
        for dataset in datasets:
            if dataset.name.split('_')[1] == val_sub: 
                #train_dataset.merge_samples(dataset)
                val_dataset = dataset
        #    else: 
        #        val_dataset = dataset
        
        # val dataset may not have all classes - replace class list with train dataset's
        train_dataset.classes={"ground_truth":classes}
        val_dataset.classes={"ground_truth":classes}

        # Load datamodules
        datamodule = VideoClassificationData.from_fiftyone(
            val_dataset=val_dataset,
            predict_dataset=val_dataset,
            batch_size=1,
            num_workers=4,
            clip_sampler="uniform",
            decode_audio=False,
        )

        modelname = next(x for x in models if (subdir + '_') in x)
        model = VideoClassifier.load_from_checkpoint(Path(models_root, modelname))

        # 3. Create the trainer
        trainer = flash.Trainer(max_epochs=2, gpus=torch.cuda.device_count())

        # 4. Make a prediction
        print(f"=== Predicting: {modelname} ===")
        predictions = trainer.predict(model, datamodule=datamodule, output="labels")

        for i, sample in enumerate(val_dataset):
            sample[modelarch+"_predictions"] = fo.Classification(label=predictions[i][0])
            sample.save()

        results = val_dataset.evaluate_classifications(
            modelarch+"_predictions", 
            gt_field="ground_truth",
            eval_key="eval"+modelarch,
            classes=val_dataset.classes,
            )
        results.print_report()
        report = results.report()

        # Save Results
        print("== Save predictions ==")
        results_dir = Path('Results', modelarch)
        os.makedirs(results_dir, exist_ok=True)
        resultspath = Path(results_dir, modelname+'.json')
        results.write_json(resultspath, pretty_print=True)

        # Save Report
        report_dir = Path('Results', modelarch+"_report")
        os.makedirs(report_dir, exist_ok=True)
        reportpath = Path(report_dir, modelname+'report'+'.json')
        with open(reportpath, 'w') as file: file.write(json.dumps(report))





if __name__ == '__main__':
    main()