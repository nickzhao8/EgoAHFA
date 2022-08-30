from pathlib import Path
import os, math

dataset_root = Path('D:/zhaon/Datasets/Video_JPG_Stack').resolve()
num_frames = 16         # Total number of frames in the clip
temporal_stride = 4     # Temporal stride within the clip
clip_length = num_frames*temporal_stride    # Total number of frames the clip spans. Divide by fps to get length in seconds.
clip_stride = int(clip_length/2)            # Number of frames between clips
annotation_filename = f'annotation_{num_frames}x{temporal_stride}.txt'

for subdir in os.listdir(dataset_root):
    f = open(Path(dataset_root,subdir,annotation_filename),'w')
    for score in os.listdir(Path(dataset_root, subdir)):
        try: int(score)
        except ValueError: continue
        for task in os.listdir(Path(dataset_root,subdir,score)):
            # new_task = task.replace(' ','_')
            # os.rename(Path(dataset_root,subdir,score,task),Path(dataset_root,subdir,score,new_task))
            frames = os.listdir(Path(dataset_root,subdir,score,task))
            total_frames = len(frames)
            num_clips = math.floor((total_frames-clip_length)/clip_stride)+1
            for i in range(num_clips):
               start_frames = i*clip_stride + 1
               end_frames = start_frames + clip_length - 1
               pathname = str(Path(score,task))
               f.write(f'{pathname} {start_frames} {end_frames} {score}\n')
    f.close()


