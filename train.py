# library
import torch
import argparse
from utils_metrics import Dice, BCEDiceLoss
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from dataloader import load_data
from model_loader import *
from utils import *
import os
from torchsummary import summary
import segmentation_models_pytorch as smp
import pandas as pd 
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BUSI',
    help='dataset name')
parser.add_argument('--split_seed', type=int, default=18,
    help='split seed')
parser.add_argument('--encoder_name', type=str, default='unet_encoder_effb4', 
    help='encoder name')  ### 'unet_encoder_effb4' or 'unet_encoder_resnet18' or 'unet_encoder_resnet34'
parser.add_argument('--segHead_name', type=str, default='unet_decoder_CSMAM',
    help='decoder name') ## 'unet_decoder_CSMAM'
parser.add_argument('--classHead_name', type=str, default='efficientnet_classifier_CA',
    help='classifier name')
parser.add_argument('--mode', type=str, default='multitask',
    help='mode of task')
parser.add_argument('--numclass', type=int, default=1,
    help='number of nodes at output layer')
parser.add_argument('--inchan', type=int, default=3,
    help='number input channels')
parser.add_argument('-e', '--epochs', type=int, default=1000,
    help='number of epochs to train our network for')
parser.add_argument('--basepath', type=str, default='../outputs/',
    help='path for saving output')
parser.add_argument('--lr', type=float, default= 1e-4,
    help='learning rate')
parser.add_argument('--alpha', type=float, default= .8,
    help='task weights')
parser.add_argument('--loss', type=str, default='BCEDiceLoss',
    help='loss function')
parser.add_argument('--aug', type=str, default=True,
    help='augmentation')
parser.add_argument('--seed', nargs = "+",type=int, default=0,
    help='seed')
parser.add_argument('--task_weight_mode', type=str, default='alpha',
    help='task weight mode')
parser.add_argument('--batchsize', type=int, default=8,
    help='batch size')
parser.add_argument('--img_ext', default='.png',
                    help='image file extension')
parser.add_argument('--mask_ext', default='.png',
                    help='mask file extension')
parser.add_argument('--input_w', default=256, type=int,
                    help='image width')
parser.add_argument('--input_h', default=256, type=int,
                    help='image height')
parser.add_argument('--wd', type=str, default=False,
    help='weigt decay')
parser.add_argument('--fold_no', default=1, type=int,
    help='fold number')
args = vars(parser.parse_args())


# train per epoch
def train(model, dataloader, mode, criterion):
    model.train()
    criterion.train()

    # print('Training started for ', mode)
    epoch_train_loss = 0.0
    if mode == 'multitask':
        epoch_train_seg_loss = 0.0
        epoch_train_class_loss = 0.0

    for i, data in enumerate(dataloader):
        if mode == 'classification':
            image, _, label = data
            label = label.to(device,dtype=torch.float32)
        elif mode == 'segmentation':
            image, mask, _ = data
            mask = mask.to(device,dtype=torch.float32)
        elif mode == 'multitask':
            image, mask, label = data
            label = label.to(device,dtype=torch.float32)
            mask = mask.to(device,dtype=torch.float32)
        image = image.to(device,dtype=torch.float32)

        # print(label)
        # print(torch.min(image))
        # plt.imshow(np.moveaxis(image[0].cpu().numpy(),0,-1))
        # plt.show()
        optimizer.zero_grad()

        # forward pass
        if mode == 'classification':
            pred_label = model(image)
        elif mode == 'segmentation':
            pred_mask = model(image)
        elif mode == 'multitask':
            latent, pred_mask, pred_label = model(image)
        # print(pred_label.dtype, label.dtype)

        # losses
        if mode == 'classification':
            loss = criterion(label, torch.squeeze(pred_label),None,None,)
        elif mode == 'segmentation':
            loss = criterion(None, None, mask, pred_mask)
        elif mode == 'multitask':
            seg_loss, class_loss, loss = criterion(label, torch.squeeze(pred_label), mask, pred_mask)

        epoch_train_loss += loss.item()
        if mode == 'multitask':
            epoch_train_class_loss += class_loss.item()
            epoch_train_seg_loss += seg_loss.item()

        # backward pass
        loss.backward()

        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_train_loss = epoch_train_loss / len(dataloader)
    if mode == 'classification' or mode == 'segmentation':
        return epoch_train_loss
    if mode == 'multitask':
        epoch_train_class_loss = epoch_train_class_loss / len(dataloader)
        epoch_train_seg_loss = epoch_train_seg_loss / len(dataloader)
        return epoch_train_loss, epoch_train_class_loss, epoch_train_seg_loss


# test per epoch
def evaluate(model, dataloader, mode, criterion):
    model.eval()
    criterion.eval()
    # print('Evaluation started for ', mode)
    epoch_val_loss = 0.0
    if mode == 'classification':
        epoch_val_class_metric = 0.0
    elif mode == 'segmentation':
        epoch_val_seg_metric = 0.0
    elif mode == 'multitask':
        epoch_val_seg_loss = 0.0
        epoch_val_class_loss = 0.0
        epoch_val_class_metric = 0.0
        epoch_val_seg_metric = 0.0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if mode == 'classification':
                image, _, label = data
                label = label.to(device,dtype=torch.float32)
            elif mode == 'segmentation':
                image, mask, _ = data
                mask = mask.to(device,dtype=torch.float32)
            elif mode == 'multitask':
                image, mask, label = data
                label = label.to(device,dtype=torch.float32)
                mask = mask.to(device,dtype=torch.float32)
            image = image.to(device,dtype=torch.float32)

            # forward pass
            if mode == 'classification':
                pred_label = model(image)
            elif mode == 'segmentation':
                pred_mask = model(image)
            elif mode == 'multitask':
                latent, pred_mask, pred_label = model(image)

            # losses
            if mode == 'classification':
                loss = criterion(label, torch.squeeze(pred_label),None,None)
            elif mode == 'segmentation':
                loss = criterion(None,None,mask, pred_mask)
            elif mode == 'multitask':
                seg_loss, class_loss, loss = criterion(label, torch.squeeze(pred_label), mask, pred_mask)

            epoch_val_loss += loss.item()
            if mode == 'multitask':
                epoch_val_class_loss += class_loss.item()
                epoch_val_seg_loss += seg_loss.item()
            
            if mode == 'classification':
                # squeeze to one channel output
                pred_label_1d = torch.tensor(torch.squeeze(F.sigmoid(pred_label))>=0.5, dtype = torch.int32)
                label_1d = torch.squeeze(label).type(torch.int32)
                # print(pred_label_1d,label_1d)

                epoch_val_class_metric += (pred_label_1d == label_1d).sum().item()
                # if i == 0:
                #     all_label = label_1d
                #     all_pred_label = pred_label_1d
                # else:
                #     all_label = torch.concatenate((all_label,label_1d),axis = 0)
                #     all_pred_label = torch.concatenate((all_pred_label,pred_label_1d),axis = 0)
            elif mode == 'segmentation':
                # print(pred_label, pred_label_1d)
                pred_mask_2d = torch.squeeze(torch.tensor(torch.sigmoid(pred_mask)>=0.5,dtype = torch.float32).clone().detach())
                mask_2d = torch.squeeze(mask)
                epoch_val_seg_metric += Dice(pred_mask_2d, mask_2d).item()
            elif mode == 'multitask':
                # squeeze to one channel output
                pred_label_1d = torch.tensor(torch.squeeze(F.sigmoid(pred_label).clone().detach())>=0.5, dtype = torch.int32)
                # pred_label_1d = torch.argmax(pred_label,dim = 1)
                label_1d = torch.squeeze(label).type(torch.int32)
                epoch_val_class_metric += (pred_label_1d == label_1d).sum().item()
                # print(pred_label, pred_label_1d)
                pred_mask_2d = torch.squeeze(torch.tensor(torch.sigmoid(pred_mask.clone().detach())>=0.5,dtype = torch.float32))
                mask_2d = torch.squeeze(mask)
                epoch_val_seg_metric += Dice(pred_mask_2d, mask_2d).item()
                # if i == 0:
                #     all_label = label_1d
                #     all_pred_label = pred_label_1d
                # else:
                #     all_label = torch.concatenate((all_label,label_1d),axis = 0)
                #     all_pred_label = torch.concatenate((all_pred_label,pred_label_1d),axis = 0)

    # loss and accuracy for the complete epoch
    epoch_val_loss = epoch_val_loss / len(dataloader)
    if mode == 'classification':
        epoch_val_class_metric = epoch_val_class_metric / len(dataloader.dataset)
        return epoch_val_loss, epoch_val_class_metric
    elif mode == 'segmentation':
        epoch_val_seg_metric = epoch_val_seg_metric / len(dataloader)
        return epoch_val_loss, epoch_val_seg_metric
    if mode == 'multitask':
        epoch_val_seg_loss = epoch_val_seg_loss / len(dataloader)
        epoch_val_class_loss = epoch_val_class_loss / len(dataloader)
        epoch_val_class_metric = epoch_val_class_metric / len(dataloader.dataset)
        epoch_val_seg_metric = epoch_val_seg_metric / len(dataloader)
        return epoch_val_loss, epoch_val_class_loss, epoch_val_seg_loss, epoch_val_class_metric,epoch_val_seg_metric


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, mode , task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.mode = mode
        self.task_num = task_num
        if self.mode == 'classification':
            self.celoss  = nn.BCEWithLogitsLoss().to(device)#nn.BCEWithLogitsLoss(pos_weight = torch.tensor([4.0])).to(device)
        elif self.mode == 'segmentation':
            self.segloss = BCEDiceLoss().to(device)
        elif self.mode =='multitask':
            self.segloss = BCEDiceLoss().to(device)
            self.celoss   = nn.BCEWithLogitsLoss().to(device)

            self.log_vars = nn.Parameter(torch.tensor((0.0,0.0),requires_grad=True)) #1.0, 6.0

    def forward(self, label, pred_label,mask, pred_mask):
        if self.mode == 'classification':
            loss = self.celoss(pred_label,torch.squeeze(label))
            return loss
        elif self.mode == 'segmentation':
            loss = self.segloss(pred_mask,mask)
            return loss
        elif self.mode == 'multitask':
            std_1 = torch.exp(self.log_vars[0]) ** 0.5
            std_2 = torch.exp(self.log_vars[1]) ** 0.5

            seg_loss = self.segloss(pred_mask,mask)
            cls_loss = self.celoss(pred_label,torch.squeeze(label))


            # loss = torch.mean(seg_loss_1+cls_loss_1)
            loss = 0.8*seg_loss + 0.2*cls_loss
            return seg_loss,cls_loss,loss


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)
        print('Current learning rate',param_group['lr'])

# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

set_seed(args['seed'][0])

# load data
# trainloader, valloader, testloader = load_data(args['dataset'], args['numclass'],args['batchsize'], args['input_h'], args['input_w'], args['img_ext'], args['mask_ext'], args['split_seed'])
trainloader, valloader, testloader = load_data(args['dataset'], args['numclass'],args['batchsize'], args['input_h'], args['input_w'], args['img_ext'], args['mask_ext'], args['split_seed'],args['fold_no'])

print('data_loaded')
# load model

if args['encoder_name'] == 'smp_unet_encoder_effb4':
    model = MultiTaskModel(args['encoder_name'], args['segHead_name'], args['classHead_name'], args['inchan'], [448,160,56,32,48], args['numclass'], args['mode']).to(device)
else:
    model = MultiTaskModel(args['encoder_name'], args['segHead_name'], args['classHead_name'], args['inchan'], [256,128,64,32,16], args['numclass'], args['mode']).to(device)


# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")


# loss function
criterion = MultiTaskLossWrapper(args['mode'],2).to(device)

# loss and optimizer
if args['wd'] == True:
    print('adam with weight decay')
    optimizer = optim.Adam([{'params': model.parameters()},{'params': criterion.parameters()}], lr=args['lr'], weight_decay = 0.001)
else:
    optimizer = optim.Adam([{'params': model.parameters()},{'params': criterion.parameters()}], lr=args['lr'])


# initialize SaveBestModel class
if args['mode'] == 'classification':
    save_best_model = SaveBestModel_class()
elif args['mode'] == 'segmentation':
    save_best_model = SaveBestModel_seg()
elif args['mode'] == 'multitask':
    save_best_model1 = SaveBestModel_class()
    save_best_model2 = SaveBestModel_seg()    
    save_best_model3 = SaveBestModel_multitask()

# initialize the dictionary
if args['mode'] == 'classification':
    log = OrderedDict([
        ('epoch', []),
        ('train_loss', []),
        ('train_class_metric', []),
        ('val_loss', []),
        ('val_class_metric', []),
        ('test_loss', []),
        ('test_class_metric', []),])
elif args['mode'] == 'segmentation':
    log = OrderedDict([
        ('epoch', []),
        ('train_loss', []),
        ('train_seg_metric', []),
        ('val_loss', []),
        ('val_seg_metric', []),
        ('test_loss', []),
        ('test_seg_metric', []),])
elif args['mode'] == 'multitask':
    log = OrderedDict([
        ('epoch', []),
        ('train_loss', []),
        ('train_seg_loss', []),
        ('train_class_loss', []),
        # ('weight1',[]),
        # ('weight2', []),
        ('train_seg_metric', []),
        ('train_class_metric', []),
        ('val_loss', []),
        ('val_seg_loss', []),
        ('val_class_loss', []),
        ('val_seg_metric', []),
        ('val_class_metric', []),
        ('test_loss', []),
        ('test_seg_loss', []),
        ('test_class_loss', []),
        ('test_seg_metric', []),
        ('test_class_metric', []),])

#savepath
savePath = args['basepath']+args['dataset']+'/'+str(args['split_seed'])+'/'+args['mode']+'_'+args['encoder_name']+'_'+args['segHead_name']+'_'+args['classHead_name']+'_'+args['loss']+'_seed_'+str(args['seed'][0])+'_epoch_'+str(args['epochs'])+'_aug_alr_00001_tw_'+str(int(10*args['alpha']))+'_'+str(args['task_weight_mode'])+'_bilinear'+'no_spatial'+str(args['fold_no'])
print(len(trainloader),len(valloader),len(testloader))
if not  os.path.exists(savePath):
    os.makedirs(savePath)

with open(savePath+'/commandline_args.txt', 'w') as f:
    json.dump(args, f, indent=2)

# start the training
mode = args['mode']
for epoch in range(500):#args['epochs']):
    # print(f"[INFO]: Epoch {epoch+1} of {args['epochs']}")
    print(f"[INFO]: Epoch {epoch+1} of 500")

    adjust_learning_rate(optimizer, epoch, args['epochs'], args['lr'], power=0.9)
    if args['mode'] == 'classification':
        train(model, trainloader, mode, criterion)
        epoch_train_loss, epoch_train_class_metric = evaluate(model, trainloader, mode, criterion)
        # print('done')
        epoch_val_loss, epoch_val_class_metric = evaluate(model, valloader, mode, criterion)
        epoch_test_loss, epoch_test_class_metric = evaluate(model, testloader, mode, criterion)

        print(f"Training classification loss: {epoch_train_loss:.3f}")
        print(f"Validation classification loss: {epoch_val_loss:.3f}")
        print(f"Testing classification loss: {epoch_test_loss:.3f}")

        save_best_model(epoch_val_class_metric, epoch, model, optimizer, criterion,savePath)

    elif args['mode'] == 'segmentation':
        train(model, trainloader, mode, criterion)
        epoch_train_loss, epoch_train_seg_metric = evaluate(model, trainloader, mode, criterion)
        epoch_val_loss, epoch_val_seg_metric = evaluate(model, valloader, mode, criterion)
        epoch_test_loss, epoch_test_seg_metric = evaluate(model, testloader, mode, criterion)

        print(f"Training segmentation loss: {epoch_train_loss:.3f}")
        print(f"Validation segmentation loss: {epoch_val_loss:.3f}")
        print(f"Testing segmentation loss: {epoch_test_loss:.3f}")

        save_best_model(epoch_val_seg_metric, epoch, model, optimizer, criterion,savePath)

    elif args['mode'] == 'multitask':
        train(model, trainloader, mode, criterion)
        epoch_train_loss, epoch_train_class_loss, epoch_train_seg_loss, epoch_train_class_metric, epoch_train_seg_metric = evaluate(model, trainloader, mode, criterion)
        epoch_val_loss, epoch_val_class_loss, epoch_val_seg_loss, epoch_val_class_metric, epoch_val_seg_metric = evaluate(model, valloader, mode, criterion)
        epoch_test_loss, epoch_test_class_loss, epoch_test_seg_loss, epoch_test_class_metric, epoch_test_seg_metric = evaluate(model, testloader, mode, criterion)

        print(f"Training loss: {epoch_train_loss:.3f}, training segmentation loss: {epoch_train_seg_loss:.3f}, training classification loss: {epoch_train_class_loss:.3f}")
        print(f"Validation loss: {epoch_val_loss:.3f}, validation segmentation loss: {epoch_val_seg_loss:.3f}, validation classification loss: {epoch_val_class_loss:.3f}")
        print(f"Testing loss: {epoch_test_loss:.3f}, testing segmentation loss: {epoch_test_seg_loss:.3f}, testing classification loss: {epoch_test_class_loss:.3f}")

        # cdict = {name:param.tolist() for name,param in criterion.named_parameters()}
        # weight1 = cdict['log_vars'][0]
        # weight2 = cdict['log_vars'][1]

        save_best_model1(epoch_val_class_metric, epoch, model, optimizer, criterion,savePath)
        save_best_model2(epoch_val_seg_metric, epoch, model, optimizer, criterion,savePath)
        save_best_model3(epoch_val_class_metric+epoch_val_seg_metric, epoch, model, optimizer, criterion,savePath)


    print('-'*50)


    if args['mode'] == 'classification':
        log['epoch'].append(epoch)
        log['train_loss'].append(epoch_train_loss)
        log['train_class_metric'].append(epoch_train_class_metric)
        log['val_loss'].append(epoch_val_loss)
        log['val_class_metric'].append(epoch_val_class_metric)
        log['test_loss'].append(epoch_test_loss)
        log['test_class_metric'].append(epoch_test_class_metric)
    elif args['mode'] == 'segmentation':
        log['epoch'].append(epoch)
        log['train_loss'].append(epoch_train_loss)
        log['train_seg_metric'].append(epoch_train_seg_metric)
        log['val_loss'].append(epoch_val_loss)
        log['val_seg_metric'].append(epoch_val_seg_metric)
        log['test_loss'].append(epoch_test_loss)
        log['test_seg_metric'].append(epoch_test_seg_metric)
    elif args['mode'] == 'multitask':
        log['epoch'].append(epoch)
        log['train_loss'].append(epoch_train_loss)
        log['train_seg_loss'].append(epoch_train_seg_loss)
        log['train_class_loss'].append(epoch_train_class_loss)
        # log['weight1'].append(weight1)
        # log['weight2'].append(weight2)
        log['train_seg_metric'].append(epoch_train_seg_metric)
        log['train_class_metric'].append(epoch_train_class_metric)
        log['val_loss'].append(epoch_val_loss)
        log['val_seg_loss'].append(epoch_val_seg_loss)
        log['val_class_loss'].append(epoch_val_class_loss)
        log['val_seg_metric'].append(epoch_val_seg_metric)
        log['val_class_metric'].append(epoch_val_class_metric)
        log['test_loss'].append(epoch_test_loss)
        log['test_seg_loss'].append(epoch_test_seg_loss)
        log['test_class_loss'].append(epoch_test_class_loss)
        log['test_seg_metric'].append(epoch_test_seg_metric)
        log['test_class_metric'].append(epoch_test_class_metric)


    pd.DataFrame(log).to_csv(savePath+'/log.csv', index=False)

# save the trained model weights for a final time
save_model(epoch, model ,optimizer, criterion,savePath)


# save the loss and accuracy plots
save_plots_multitask(log , savePath,args['mode'])
print('TRAINING COMPLETE')
