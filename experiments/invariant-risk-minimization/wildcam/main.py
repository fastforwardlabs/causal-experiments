import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import random

from dataset import get_dataset, get_handler
from models import get_net
from train import Train
from torchvision import transforms

def get_accuracy(predictions, y):
    return 1.0 * (y == predictions).sum().item() / len(y)

if __name__ == "__main__":
    
    seed = 123
    dataset_name = 'WILDCAM'
    
    args_pool = {
        'WILDCAM': {
            'n_epoch': 10,
            'n_classes': 2,
            'fc_only': True,
            'model_path': "/home/shioulin/active-learning/deep-active/caltech_model.pth",
            'transform': {
                'train': transforms.Compose([transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'test': transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
            },
            'loader_tr_args': {
                'batch_size': 100, 
                'num_workers': 1
            },
            'loader_te_args': {
                'batch_size': 100, 
                'num_workers': 1
            },
            'loader_sample_args': {
                'batch_size': 100, 
                'num_workers': 1
            },
            'optimizer_args': {
                'lr': 0.001,
                'l2_regularizer_weight': 0.001,
                'penalty_anneal_iters': 2,
                'penalty_weight': 10.0
            },
            'mode': 'IRM'  #training mode - IRM or ERM
        }
    }
       
    args = args_pool[dataset_name]

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # load dataset
    x_train, y_train, x_test, y_test = get_dataset(dataset_name)
    print("x_train: ", len(x_train))
    print("y_train: ", y_train.shape)
    print("x_test: ", len(x_test))
    print("y_test: ", y_test.shape)
    
    # get model
    net = get_net(dataset_name)
    handler = get_handler(dataset_name)
    #torch.backends.cudnn.enabled = True
    print("Using GPU - {}".format(torch.cuda.is_available()))
    train_process = Train(x_train, y_train, x_test, y_test, net, handler, args)
    train_process.train()
