import os
import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob
from albumentations.augmentations.transforms import Normalize
from albumentations.augmentations.geometric.transforms import Flip
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90,Resize
import csv
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import StratifiedKFold

class DatasetHAM10000(torch.utils.data.Dataset):
    def __init__(self, img_ids, cls_ids,img_dir, mask_dir, img_ext, mask_ext, num_classes,  transform=None):
   
        self.img_ids = img_ids
        self.cls_ids = cls_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        
        img_id = self.img_ids[idx]
        cls_id = self.cls_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = []
        # mask.append(cv2.imread(os.path.join(self.mask_dir,img_id +self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask.append(cv2.imread(os.path.join(self.mask_dir,img_id +'_segmentation' +self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        # print(mask.shape)
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, cls_id #, {'img_id': img_id}#torch.nn.functional.one_hot(torch.tensor(self.dict[img_id]), num_classes=3)#



class DatasetBUSI(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        
        img_id = self.img_ids[idx]
        class_name = img_id.split(" ")[0]
        if class_name  == 'benign':
            class_id = 0
        elif class_name  == 'malignant':
            class_id = 1
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = []
        # mask.append(cv2.imread(os.path.join(self.mask_dir,img_id +self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask.append(cv2.imread(os.path.join(self.mask_dir,img_id +'_mask' +self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        # print(mask.shape)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, class_id #, {'img_id': img_id}







def load_data(dataset,num_classes,batchsize,input_h,input_w,img_ext,mask_ext,split_seed,k):
    if dataset == 'BUSI':
        # Data loading code
        inputPath = '../inputs'
        # num_classes = 3
        img_ids = glob(os.path.join(inputPath, dataset, 'images', '*' + img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        cls_id = []
        for img_id in img_ids:
            class_name = img_id.split(" ")[0]
            if class_name  == 'benign':
                cls_id.append(0)
            elif class_name  == 'malignant':
                cls_id.append(1)
        # print(cls_id.count(0),cls_id.count(1))

        skf = StratifiedKFold(n_splits=5, random_state=split_seed, shuffle=True)
        train_img_index, test_img_index = list(skf.split(img_ids,cls_id))[k-1]
        train_img_ids = [img_ids[i] for i in train_img_index]
        test_img_index = [img_ids[i] for i in test_img_index]


        train_transform = Compose([
            RandomRotate90(),
            Flip(),
            Resize(input_h, input_w),
            Normalize(),
        ])

        test_transform = Compose([
            Resize(input_h, input_w),
            Normalize(),
        ])

        train_dataset = DatasetBUSI(
            img_ids=train_img_ids,
            img_dir=os.path.join(inputPath, dataset, 'images'),
            mask_dir=os.path.join(inputPath, dataset, 'masks'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=num_classes,
            transform=train_transform)

        test_dataset = DatasetBUSI(
            img_ids=test_img_ids,
            img_dir=os.path.join(inputPath, dataset,'images'),
            mask_dir=os.path.join(inputPath, dataset,'masks'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=num_classes,
            transform=test_transform)


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batchsize,
            shuffle=True,
            drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batchsize,
            shuffle=False,
            drop_last=False)

    if dataset == 'HAM10000':
        # Data loading code
        inputPath = '/home/mlrl/Susmita/Multitask/morpho-attention/inputs'
        inputCSVPath = '/home/mlrl/Susmita/raw_data/segmentation/HAM10000/HAM10000_metadata.csv'
        with open(inputCSVPath, newline='') as file: 
            reader = csv.reader(file, delimiter = ' ') 
            headings = next(reader)  
            Output = [] 
            mydict1 = {}
            mydict2 = {}

            count = 0
            for row in reader:
                key = row[0].split(',')[0]
                value = row[0].split(',')[1]
                subs = row[0].split(',')[0]
                mydict1.setdefault(key,[]).append(value)
                value = row[0].split(',')[2]
                if row[0].split(',')[2] == 'akiec':
                    class_id = 0
                elif row[0].split(',')[2] == 'bcc':
                    class_id = 1
                elif row[0].split(',')[2] == 'bkl':
                    class_id = 2
                elif row[0].split(',')[2] == 'df':
                    class_id = 3
                elif row[0].split(',')[2] == 'mel':
                    class_id = 4
                elif row[0].split(',')[2] == 'nv':
                    class_id = 5
                elif row[0].split(',')[2] == 'vasc':
                    class_id = 6
                mydict2.update({key:class_id})

                # pause()

        # print(mydict2)
        img_ids = []
        cls_id = []
        img_ids = [key for key in mydict2]
        cls_id = [mydict2[key] for key in mydict2]
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        # k = 2
        train_sub_ids, val_sub_ids = list(skf.split(img_ids, cls_id))[k-1]
        train_sub = [img_ids[i] for i in train_sub_ids]
        val_sub = [img_ids[i] for i in val_sub_ids]
        train_cls_id = [cls_id[i] for i in train_sub_ids]
        val_cls_id = [cls_id[i] for i in val_sub_ids]

        train_img_ids = []
        train_cls_ids = []
        for subs,cls_id in zip(train_sub,train_cls_id):
            train_img_ids.extend(mydict1[subs])
            train_cls_ids.extend(len(mydict1[subs])*[cls_id])

        val_img_ids = []
        val_cls_ids = []
        for subs,cls_id in zip(val_sub,val_cls_id):
            val_img_ids.extend(mydict1[subs])
            val_cls_ids.extend(len(mydict1[subs])*[cls_id])

        test_img_ids = val_img_ids 
        test_cls_ids = val_cls_ids 

        train_transform = Compose([
            RandomRotate90(),
            Flip(),
            # Resize(input_h, input_w),
            Normalize(),
        ])

        val_transform = Compose([
            # Resize(input_h, input_w),
            Normalize(),
        ])

        train_dataset = DatasetHAM10000(
            img_ids=train_img_ids,
            cls_ids=train_cls_ids,
            img_dir=os.path.join(inputPath, dataset, 'images'),
            mask_dir=os.path.join(inputPath, dataset, 'masks'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=num_classes,
            transform=train_transform)
        val_dataset = DatasetHAM10000(
            img_ids=val_img_ids,
            cls_ids=val_cls_ids,
            img_dir=os.path.join(inputPath, dataset,'images'),
            mask_dir=os.path.join(inputPath, dataset,'masks'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=num_classes,
            transform=val_transform)
        test_dataset = DatasetHAM10000(
            img_ids=test_img_ids,
            cls_ids=test_cls_ids,
            img_dir=os.path.join(inputPath, dataset,'images'),
            mask_dir=os.path.join(inputPath, dataset,'masks'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=num_classes,
            transform=val_transform)


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batchsize,
            shuffle=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize,
            shuffle=False,
            drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batchsize,
            shuffle=False,
            drop_last=False)

    return train_loader, test_loader
