# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:31:48 2022

@author: Meng-Fen Tsai
"""

import torch
import gzip
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

class HandRoleDataset(Dataset):
    """ load enture dataset in the GPU memory to speed up training """
    """ get filename, hand_id, handrole feature, label """
    def __init__(self, image_path, labeling_path):
        super(HandRoleDataset, self).__init__()
        self.image_path=image_path;
        self.labeling_path=labeling_path;   
        self.imagename=[];
        self.hand_info=[];       
        self.handrole_fea=[];
        self.labels=[]; 
        #### Redundent codes were deleted.
                    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):      
        imagename=self.imagename[idx];
        label=self.labels[idx];
        handrole_fea=self.handrole_fea[idx];  
        hand_info=self.hand_info[idx];        
        
        summary=[imagename, hand_info, handrole_fea, label];
        
        return summary

## get the dataset ready
image_list=['image_path_1', 'image_path2',...];
dataloader_folder='/mnt/wwn-0x5000c500dc5d7cdc-part2/DataLoader_HandRole/';
labeling_txt_path='LOSOCV_Home_HomeLab_sub1_Manipulation_train.txt';
dataset=HandRoleDataset(image_path=image_list, labeling_path=labeling_txt_path);

## save the unshuffled dataset for now (to save time).
start_loading=datetime.now();
torch.save(dataset, gzip.GzipFile('save_loader_name.gz','wb')); 
end_loading=datetime.now();
diff_time=(end_loading-start_loading).total_seconds();#in econds
print('Saving the zipped dataset takes: %.2f minutes' % (diff_time/60))

## load the saved dataset and shuffle it after loading it.
start_loading=datetime.now();
train_dataset=torch.load(gzip.GzipFile('save_loader_name.gz','rb'));
train_loader=DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0);# shuffle the dataset after you load it, it will be much more faster.
del train_dataset #clear some memory here
torch.cuda.empty_cache()### please add this time to actually clean up the memory
end_loading=datetime.now();
diff_time=(end_loading-start_loading).total_seconds();#in econds
print('Loading the zipped dataset takes: %.2f minutes' % (diff_time/60))
