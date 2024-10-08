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
        test_img_ids = [img_ids[i] for i in test_img_index]


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

    if dataset == 'UDIAT':
        # Data loading code
        inputPath = '/home/mlrl/Susmita/Multitask/morpho-attention/inputs'
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
        test_img_ids = [img_ids[i] for i in test_img_index]



        train_transform = Compose([
            RandomRotate90(),
            Flip(),
            Resize(input_h, input_w),
            Normalize(),
        ])

        val_transform = Compose([
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
            transform=val_transform)


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



    return train_loader, test_loader
