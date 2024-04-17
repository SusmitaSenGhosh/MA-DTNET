import torch.nn as nn


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


import torch
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion,path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,

                }, path+'/final_model.pth')


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss,epoch, model, optimizer, criterion,path
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path+'/best_model.pth')

class SaveBestModel_class:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_class_metric=float('-inf')
    ):
        self.best_valid_class_metric = best_valid_class_metric
        
    def __call__(
        self, current_valid_all_metric,epoch, model, optimizer, criterion,path,
    ):
        if current_valid_all_metric > self.best_valid_class_metric:
            self.best_valid_class_metric = current_valid_all_metric
            print(f"\nBest validation class metric: {self.best_valid_class_metric}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': criterion,
                }, path+'/best_model_class.pth')

class SaveBestModel_seg:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_seg_metric=float('-inf')
    ):
        self.best_valid_seg_metric = best_valid_seg_metric
        
    def __call__(
        self, current_valid_seg_metric,epoch, model, optimizer, criterion,path,
    ):
        if current_valid_seg_metric > self.best_valid_seg_metric:
            self.best_valid_seg_metric = current_valid_seg_metric
            print(f"\nBest validation seg metric: {self.best_valid_seg_metric}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path+'/best_model_seg.pth')

class SaveBestModel_multitask:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_all_metric=float('-inf')
    ):
        self.best_valid_all_metric = best_valid_all_metric
        
    def __call__(
        self, current_valid_all_metric,epoch, model, optimizer, criterion,path,
    ):
        if current_valid_all_metric > self.best_valid_all_metric:
            self.best_valid_all_metric = current_valid_all_metric
            print(f"\nBest validation multitask metric: {self.best_valid_all_metric}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path+'/best_model_multitask.pth')

def save_plots(train_acc, valid_acc, test_acc,train_loss, valid_loss,test_loss,path):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.plot(
        test_acc, color='red', linestyle='-', 
        label='test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path+'/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='green', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='blue', linestyle='-', 
        label='validataion loss'
    )
    plt.plot(
        test_loss, color='red', linestyle='-', 
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+'/loss.png')


def set_seed(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss





def save_plots_multitask(log, path,mode):
    """
    Function to save the loss and accuracy plots to disk.
    """
    if mode == 'multitask':
        # seg loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_seg_loss'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_seg_loss'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_seg_loss'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('segmentation loss')
        plt.legend()
        plt.savefig(path+'/seg_loss.png')
        
        # class loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_class_loss'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_class_loss'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_class_loss'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('classification loss')
        plt.legend()
        plt.savefig(path+'/class_loss.png')

        # seg metric plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_seg_metric'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_seg_metric'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_seg_metric'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('segmentation metric')
        plt.legend()
        plt.savefig(path+'/seg_metric.png')

        # class metric plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_class_metric'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_class_metric'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_class_metric'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('classification metric')
        plt.legend()
        plt.savefig(path+'/class_metric.png')
    elif mode == 'segmentation':
        # seg loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_loss'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_loss'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_loss'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('segmentation loss')
        plt.legend()
        plt.savefig(path+'/seg_loss.png')

        # seg metric plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_seg_metric'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_seg_metric'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_seg_metric'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('segmentation metric')
        plt.legend()
        plt.savefig(path+'/seg_metric.png')

    elif mode == 'classification':
        # class loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_loss'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_loss'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_loss'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('classification loss')
        plt.legend()
        plt.savefig(path+'/class_loss.png')

        # class metric plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            log['train_class_metric'], color='green', linestyle='-', 
            label='train '
        )
        plt.plot(
            log['val_class_metric'], color='blue', linestyle='-', 
            label='validataion '
        )
        plt.plot(
            log['test_class_metric'], color='red', linestyle='-', 
            label='test '
        )
        plt.xlabel('Epochs')
        plt.ylabel('classification metric')
        plt.legend()
        plt.savefig(path+'/class_metric.png')
