# library
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
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
from sklearn import metrics
import seg_metrics.seg_metrics as sg
from sklearn.metrics import confusion_matrix


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='UDIAT',
    help='dataset name')
parser.add_argument('--split_seed', type=int, default=18,
    help='split seed')
parser.add_argument('--encoder_name', type=str, default='unet_encoder_resnet34',
    help='encoder name')
parser.add_argument('--segHead_name', type=str, default='unet_decoder_resnet_CSMAM',
    help='decoder name')
parser.add_argument('--classHead_name', type=str, default='resnet_classifier_CA',
    help='classifier name')
parser.add_argument('--mode', type=str, default='multitask',
    help='mode of task')
parser.add_argument('--numclass', type=int, default=1,
    help='number of nodes at output layer')
parser.add_argument('--inchan', type=int, default=3,
    help='number input channels')
parser.add_argument('-e', '--epochs', type=int, default=1000,
    help='number of epochs to train our network for')
parser.add_argument('--basepath', type=str, default='./outputs/',
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
parser.add_argument('--fold_no', nargs = "+",type=int, default=1,
                    help='fold number')
args = vars(parser.parse_args())



# test per epoch
def evaluate(model, dataloader, mode, criterion,eval_mode):
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

            # # losses
            # if mode == 'classification':
            #     loss = criterion(label, torch.squeeze(pred_label),None,None)
            # elif mode == 'segmentation':
            #     loss = criterion(None,None,mask, pred_mask)
            # elif mode == 'multitask':
            #     seg_loss, class_loss, loss = criterion(label, torch.squeeze(pred_label), mask, pred_mask)

            # epoch_val_loss += loss.item()
            # if mode == 'multitask':
            #     epoch_val_class_loss += class_loss.item()
            #     epoch_val_seg_loss += seg_loss.item()
            
            if eval_mode == 'classification':
                # squeeze to one channel output
                # pred_label_1d = torch.squeeze(F.sigmoid(pred_label))
                # pred_label_bin_1d = torch.tensor(torch.squeeze(F.sigmoid(pred_label))>=0.5, dtype = torch.int32)
                # label_1d = torch.squeeze(label,dim = 0).type(torch.int32)
                pred_label_1d = F.sigmoid(pred_label)
                pred_label_bin_1d = torch.tensor(F.sigmoid(pred_label)>=0.5, dtype = torch.int32)
                label_1d = label.type(torch.int32)
                # print(pred_label_bin_1d,label_1d)
                # epoch_val_class_metric += (pred_label_bin_1d == label_1d).sum().item()
                # print(epoch_val_class_metric)

                # print(label.shape,label_1d.shape)

                if i == 0:
                    all_pred_label = pred_label_1d
                    all_pred_label_bin = pred_label_bin_1d
                    all_label = label_1d
                else:
                    all_label = torch.concatenate((all_label,label_1d),axis = 0)
                    all_pred_label = torch.concatenate((all_pred_label,pred_label_1d),axis = 0)
                    all_pred_label_bin = torch.concatenate((all_pred_label_bin,pred_label_bin_1d),axis = 0)

                # print(all_label.shape)
                # print(all_pred_label.shape)
                # print(all_pred_label_bin.shape)


                # if i == 0:
                #     all_label = label_1d
                #     all_pred_label = pred_label_1d
                # else:
                #     all_label = torch.concatenate((all_label,label_1d),axis = 0)
                #     all_pred_label = torch.concatenate((all_pred_label,pred_label_1d),axis = 0)
            elif eval_mode == 'segmentation':
                # print(pred_label, pred_label_1d)
                # pred_mask_2d = torch.squeeze(torch.tensor(torch.sigmoid(pred_mask)>=0.5,dtype = torch.float32).clone().detach())
                # mask_2d = torch.squeeze(mask)
                pred_mask_2d = torch.tensor(torch.sigmoid(pred_mask)>=0.5,dtype = torch.float32).clone().detach()
                mask_2d = mask
                if i == 0:
                    all_pred_mask = pred_mask_2d
                    all_mask = mask_2d
                else:
                    all_pred_mask = torch.concatenate((all_pred_mask,pred_mask_2d),axis = 0)
                    all_mask = torch.concatenate((all_mask,mask_2d),axis = 0)

                # print(mask_2d.shape)
                # print(pred_mask_2d.shape)

                # print(all_mask.shape)
                # print(all_pred_mask.shape)
                
            elif eval_mode == 'multitask':
                # squeeze to one channel output
                pred_label_1d = torch.tensor(torch.squeeze(F.sigmoid(pred_label).clone().detach())>=0.5, dtype = torch.int32)
                # pred_label_1d = torch.argmax(pred_label,dim = 1)
                label_1d = torch.squeeze(label).type(torch.int32)
                epoch_val_class_metric += (pred_label_1d == label_1d).sum().item()
                # print(pred_label, pred_label_1d)
                pred_mask_2d = torch.squeeze(torch.tensor(torch.sigmoid(pred_mask.clone().detach())>=0.5,dtype = torch.float32))
                mask_2d = torch.squeeze(mask)
                epoch_val_seg_metric += Dice(pred_mask_2d, mask_2d).item()
                if i == 0:
                    all_label = label_1d
                    all_pred_label = pred_label_1d
                else:
                    all_label = torch.concatenate((all_label,label_1d),axis = 0)
                    all_pred_label = torch.concatenate((all_pred_label,pred_label_1d),axis = 0)

    # loss and accuracy for the complete epoch
    # epoch_val_loss = epoch_val_loss / len(dataloader)
    if eval_mode == 'classification':
        epoch_val_class_acc  = (torch.squeeze(all_pred_label_bin) == all_label).sum().item()/len(dataloader.dataset)
        epoch_val_precision = metrics.precision_score(all_label.cpu(), all_pred_label_bin.cpu(), pos_label=1)
        epoch_val_recall = metrics.recall_score(all_label.cpu(), all_pred_label_bin.cpu(), pos_label=1)
        epoch_val_f1score = metrics.f1_score(all_label.cpu(), all_pred_label_bin.cpu(), pos_label=1)
        fpr, tpr, thresholds = metrics.roc_curve(all_label.cpu(), all_pred_label.cpu(), pos_label=1)
        epoch_val_class_auroc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(all_label.cpu(), all_pred_label_bin.cpu()).ravel()
        epoch_val_spec = tn / (tn+fp)
        # print(fpr,tpr,epoch_val_class_auroc)
        # epoch_val_class_metric = epoch_val_class_metric / len(dataloader.dataset)
        # print(epoch_val_class_acc, epoch_val_class_auroc)
        # print(epoch_val_class_metric)
        return  epoch_val_class_acc, epoch_val_precision, epoch_val_recall, epoch_val_spec, epoch_val_f1score, epoch_val_class_auroc
    elif eval_mode == 'segmentation':
        dice = 0
        hd = 0
        sg_dice = 0
        sg_jaccard = 0
        sg_precision = 0
        sg_recall = 0
        sg_fpr = 0
        sg_fnr = 0
        sg_spec = 0
        sg_acc = 0
        cc= 0
        for j in range(0,all_mask.shape[0]):
            mask = torch.squeeze(all_mask[j])
            pred = torch.squeeze(all_pred_mask[j])
            # print(mask,pred)
            dice += Dice(pred, mask).item()
            labels = [1]
            seg_metrics = sg.write_metrics(labels=labels,  # exclude background if needed
                  gdth_img=mask.cpu().detach().numpy(),
                  pred_img=pred.cpu().detach().numpy(),
                #   csv_file=csv_file,
                #   spacing=spacing,
                  verbose = False,
                  metrics=['dice','jaccard','precision','recall','fpr','fnr', 'hd95'],
                  TPTNFPFN=True)
            # print(len(np.unique(pred.cpu().detach().numpy())))
            # print(metrics[0]["hd"][0])    
            sg_dice += seg_metrics[0]["dice"][0]
            sg_jaccard += seg_metrics[0]["jaccard"][0]
            sg_precision += seg_metrics[0]["precision"][0]
            sg_recall += seg_metrics[0]["recall"][0]
            sg_fpr += seg_metrics[0]["fpr"][0]
            sg_fnr += seg_metrics[0]["fnr"][0]
            tp,tn,fp,fn = seg_metrics[0]['TP'][0],seg_metrics[0]['TN'][0],seg_metrics[0]['FP'][0],seg_metrics[0]['FN'][0]
            sg_spec += tn/(tn+fp)
            sg_acc += (tn+tp)/(tn+tp+fn+fp)
            # print(tp/(tp+fn), seg_metrics[0]["recall"][0])
            # print(seg_metrics[0]["fpr"],seg_metrics[0]["fnr"])
            if len(np.unique(pred.cpu().detach().numpy())) == 2:
                hd += seg_metrics[0]["hd95"][0]
                cc +=1
        # print('cc: ',cc)
        epoch_val_seg_dice = dice / len(dataloader.dataset)
        epoch_val_seg_sg_dice = sg_dice / len(dataloader.dataset)
        epoch_val_seg_sg_jaccrad = sg_jaccard / len(dataloader.dataset)
        epoch_val_seg_sg_precision = sg_precision / len(dataloader.dataset)
        epoch_val_seg_sg_recall = sg_recall / len(dataloader.dataset)
        epoch_val_seg_sg_fpr = sg_fpr / len(dataloader.dataset)
        epoch_val_seg_sg_fnr = sg_fnr / len(dataloader.dataset)
        epoch_val_seg_sg_spec = sg_spec / len(dataloader.dataset)
        epoch_val_seg_sg_acc = sg_acc / len(dataloader.dataset)

        epoch_val_seg_hd = hd / cc

        # print(epoch_val_seg_dice,epoch_val_seg_sg_dice,epoch_val_seg_hd)
        return  epoch_val_seg_dice,epoch_val_seg_sg_jaccrad,epoch_val_seg_sg_precision,epoch_val_seg_sg_recall,epoch_val_seg_sg_fpr,epoch_val_seg_sg_fnr,epoch_val_seg_hd, epoch_val_seg_sg_spec,epoch_val_seg_sg_acc
        
    # if eval_mode == 'multitask':
    #     epoch_val_seg_loss = epoch_val_seg_loss / len(dataloader)
    #     epoch_val_class_loss = epoch_val_class_loss / len(dataloader)
    #     epoch_val_class_metric = epoch_val_class_metric / len(dataloader.dataset)
    #     epoch_val_seg_metric = epoch_val_seg_metric / len(dataloader)
    #     return epoch_val_loss, epoch_val_class_loss, epoch_val_seg_loss, epoch_val_class_metric,epoch_val_seg_metric


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

            self.log_vars = nn.Parameter(torch.tensor((1.0,1.0),requires_grad=True)) #1.0, 6.0

    def forward(self, label, pred_label,mask, pred_mask):
        if self.mode == 'classification':
            loss = self.celoss(pred_label,torch.squeeze(label))
            return loss
        elif self.mode == 'segmentation':
            loss = self.segloss(pred_mask,mask)
            return loss
        elif self.mode == 'multitask':
            if args['task_weight_mode'] == 'alpha':
                seg_loss = self.segloss(pred_mask,mask)
                cls_loss = self.celoss(pred_label,torch.squeeze(label))
                loss = args['alpha']*seg_loss + (1-args['alpha'])*cls_loss
                return seg_loss,cls_loss,loss
            elif args['task_weight_mode'] == 'train':
                seg_loss = self.segloss(pred_mask,mask)
                cls_loss = self.celoss(pred_label,torch.squeeze(label))
                std_1 = torch.exp(self.log_vars[0]) ** 0.5
                std_2 = torch.exp(self.log_vars[1]) ** 0.5
                seg_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[0]) * seg_loss + self.log_vars[0],-1) #
                cls_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[1]) * cls_loss + self.log_vars[1],-1)
                loss = torch.mean(seg_loss_1+cls_loss_1)
                return seg_loss,cls_loss,loss


# computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskModel(args['encoder_name'], args['segHead_name'], args['classHead_name'], args['inchan'], [16,32,64,128,256], args['numclass'], args['mode']).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")
criterion = MultiTaskLossWrapper(args['mode'],2).to(device)

mymetrics = []
for seed,fold in zip(args['seed'],args['fold_no']):
    # set_seed(args['seed'][0])

    # load data
    _, testloader = load_data(args['dataset'], args['numclass'],args['batchsize'], args['input_h'], args['input_w'], args['img_ext'], args['mask_ext'], args['split_seed'],fold)


    #loadpath
    loadPath = args['basepath']+args['dataset']+'/'+str(args['split_seed'])+'/'+'ablation_'+args['mode']+'_'+args['encoder_name']+'_'+args['segHead_name']+'_'+args['classHead_name']+'_'+args['loss']+'_seed_'+str(seed)+'_epoch_'+str(args['epochs'])+'_aug_alr_00001_tw_'+str(int(10*args['alpha']))+'_'+str(args['task_weight_mode'])+'_bilinear_fold'+str(fold)

    # loadPath = args['basepath']+args['dataset']+'/'+str(args['split_seed'])+'/'+'ablation_'+args['mode']+'_'+args['encoder_name']+'_'+args['segHead_name']+'_'+args['classHead_name']+'_'+args['loss']+'_seed_'+str(args['seed'][0])+'_epoch_'+str(args['epochs'])+'_aug_alr_00001_tw_'+str(int(10*args['alpha']))+'_'+str(args['task_weight_mode'])+'_bilinear_fold'+str(args['fold_no'])


    mode = args['mode']

    if args['mode'] == 'classification':
        checkpoint = torch.load(loadPath+'/best_model_class.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        class_metric_acc, class_metric_precision, class_metric_recall, class_metric_spec, class_metric_f1score, class_metric_auroc =  evaluate(model, testloader, mode, criterion,mode)
        print(round(100*class_metric_acc,2),round(100*class_metric_precision,2),round(100*class_metric_recall,2),round(100*class_metric_spec,2),round(100*class_metric_f1score,2),round(class_metric_auroc,4))
        mymetrics.append([round(100*class_metric_acc,2),round(100*class_metric_precision,2),round(100*class_metric_recall,2),round(100*class_metric_spec,2),round(100*class_metric_f1score,2),round(class_metric_auroc,4)])

    elif args['mode'] == 'segmentation':
        checkpoint = torch.load(loadPath+'/best_model_seg.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        seg_metric_dice, seg_metric_jaccard,seg_metric_precision,seg_metric_recall,seg_metric_fpr,seg_metric_fnr,seg_metric_hd, seg_metric_spec,seg_metric_acc = evaluate(model, testloader, mode, criterion,mode)
        print(round(100*seg_metric_dice,2),round(100*seg_metric_jaccard,2),round(100*seg_metric_precision,2),round(100*seg_metric_recall,2),round(100*seg_metric_fpr,2),round(100*seg_metric_fnr,2),round(seg_metric_hd,2),round(100*seg_metric_spec,2),round(100*seg_metric_acc,2))
        mymetrics.append([round(100*seg_metric_dice,2),round(100*seg_metric_jaccard,2),round(100*seg_metric_precision,2),round(100*seg_metric_recall,2),round(100*seg_metric_fpr,2),round(100*seg_metric_fnr,2),round(seg_metric_hd,2),round(100*seg_metric_spec,2),round(100*seg_metric_acc,2)])

    elif args['mode'] == 'multitask':
        checkpoint = torch.load(loadPath+'/best_model_seg.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        seg_metric_dice, seg_metric_jaccard,seg_metric_precision,seg_metric_recall,seg_metric_fpr,seg_metric_fnr,seg_metric_hd, seg_metric_spec,seg_metric_acc   = evaluate(model, testloader, 'multitask', criterion, 'segmentation')


        checkpoint = torch.load(loadPath+'/best_model_class.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        class_metric_acc, class_metric_precision, class_metric_recall, class_metric_spec, class_metric_f1score, class_metric_auroc =  evaluate(model, testloader, 'multitask', criterion,'classification')


        print(round(100*seg_metric_dice,2),round(100*seg_metric_jaccard,2),round(100*seg_metric_precision,2),round(100*seg_metric_recall,2),round(100*seg_metric_fpr,2),round(100*seg_metric_fnr,2),round(seg_metric_hd,2),round(100*seg_metric_spec,2),round(100*seg_metric_acc,2),round(100*class_metric_acc,2),round(100*class_metric_precision,2),round(100*class_metric_recall,2),round(100*class_metric_spec,2),round(100*class_metric_f1score,2),round(class_metric_auroc,4))
        mymetrics.append([round(100*seg_metric_dice,2),round(100*seg_metric_jaccard,2),round(100*seg_metric_precision,2),round(100*seg_metric_recall,2),round(100*seg_metric_fpr,2),round(100*seg_metric_fnr,2),round(seg_metric_hd,2),round(100*seg_metric_spec,2),round(100*seg_metric_acc,2),round(100*class_metric_acc,2),round(100*class_metric_precision,2),round(100*class_metric_recall,2),round(100*class_metric_spec,2),round(100*class_metric_f1score,2),round(class_metric_auroc,4)])

print(np.mean(np.array(mymetrics),axis = 0))
print('EVALUATION COMPLETE')
