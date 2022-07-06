import os
from pathlib import Path
import shutil

source_root = Path('Video Segments')
subdirs = os.listdir(source_root)
new_classes = ['0','1','2']
cls_map = {'0':'0','1':'0','2':'0','3':'1','4':'2','5':'2'}
'''
Consolidated Class Mapping:
Old Class           New Class
---------------------------------------------
0               ->  0
1               ->  0
2               ->  0
3               ->  1
4               ->  2
5               ->  2

Consolidated Class Descriptions:
0 = Could not complete task.
1 = Completed task without using expected grasp/using compensatory grasp.
2 = Completed task with expected grasp.
'''

for subdir in subdirs:
    dest_root = Path('Consolidated Video Segments')
    os.makedirs(Path(dest_root, subdir), exist_ok=True)
    for cls in new_classes: os.makedirs(Path(dest_root, subdir, cls), exist_ok=True)

    classes = os.listdir(Path(source_root, subdir))
    for cls in classes:
        dest_cls = cls_map[cls]
        samples = os.listdir(Path(source_root, subdir, cls))
        for sample in samples:
            source = Path(source_root, subdir, cls, sample)
            dest = Path(dest_root, subdir, dest_cls)
            shutil.copy2(source, dest)

