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
            'n_restarts': 1,
            'steps': 50,
            'n_classes': 2,
            'fc_only': False,
            'model_path': "./models/wildcam_IRM_finetune.pth",
            'transform': {
                'train': transforms.Compose([
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'test': transforms.Compose([
                                            transforms.CenterCrop(224),
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
                'penalty_anneal_iters': 10, # make this 0 for ERM
                'penalty_weight': 1.5 # make this 0 for ERM
            }
        }
    }
       
    args = args_pool[dataset_name]
    print("\n")
    if args['optimizer_args']['penalty_weight'] > 1.0:
        print("====================IRM====================")
    else:
        print("====================ERM====================")
    print(args)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
   
    # load dataset
    x_train, y_train, x_test, y_test = get_dataset(dataset_name)
    
    print("x_train: ", len(x_train))
    print("y_train: ", y_train.shape)
    print("x_test: ", len(x_test))
    print("y_test: ", y_test.shape)
    
    #resize_img = transforms.Compose([transforms.Resize((256, 256))])    

    # get model and data handler
    net = get_net(dataset_name)
    handler = get_handler(dataset_name)
    
    # GPU enabled?
    print("Using GPU - {}".format(torch.cuda.is_available()))
    
    final_train_accs = []
    final_test_accs = []
    train_process = Train(x_train, y_train, x_test, y_test, net, handler, args)
    for restart in range(args['n_restarts']):        
        train_acc, test_acc = train_process.train()
        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)
            
        print('Final train acc (mean/std across restarts so far):')
        print(round(np.mean(final_train_accs), 4), round(np.std(final_train_accs), 4))
        print('Final test acc (mean/std across restarts so far):')
        print(round(np.mean(final_test_accs), 4), round(np.std(final_test_accs), 4))
    torch.save(train_process.clf.state_dict(), args['model_path'])
