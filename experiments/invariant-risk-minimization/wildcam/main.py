import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import random

from dataset import get_dataset, get_handler
from models import get_net
from train import Train
from torchvision import transforms

if __name__ == "__main__":
    
    seed = 123
    dataset_name = 'WILDCAM'
    
    args_pool = {
        'WILDCAM': {
            'n_restarts': 1,
            'steps': 51,
            'n_classes': 2,
            'fc_only': True,
            'model_path': "./models/",
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
                'penalty_anneal_iters': 0, # make this 0 for ERM
                'penalty_weight': 0.0 # make this 0 for ERM
            }
        }
    }
       
    args = args_pool[dataset_name]
    model_name = args['model_path'] + "wildcam_"
    print("\n")
    if args['optimizer_args']['penalty_weight'] > 1.0:
        model_name = model_name + "IRM.pth"
        print("========================================IRM========================================")
    else:
        model_name = model_name + "ERM.pth"
        print("========================================ERM========================================")
    print(args)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
   
    # load dataset
    envs, x_test, y_test = get_dataset(dataset_name)
    print("working with", len(envs), "training environments datasets: ")
    for env in envs:
        print("env['images']: ", len(env['images']))
        print("env['labels']: ", env['labels'].shape[0])
    
    print("x_test: ", len(x_test))
    print("y_test: ", y_test.shape[0])

    # get model and data handler
    net = get_net(dataset_name)
    handler = get_handler(dataset_name)
    
    # GPU enabled?
    print("Using GPU - {}".format(torch.cuda.is_available()))
    
    final_train_accs = []
    final_test_accs = []
    train_process = Train(envs, x_test, y_test, net, handler, args)
    for restart in range(args['n_restarts']):  
        print("Restart", restart)
        train_acc, test_acc = train_process.train()
        final_train_accs.append(train_acc)
        final_test_accs.append(test_acc)
            
        print('Final train acc (mean/std across restarts so far):')
        print(round(np.mean(final_train_accs), 4), round(np.std(final_train_accs), 4))
        print('Final test acc (mean/std across restarts so far):')
        print(round(np.mean(final_test_accs), 4), round(np.std(final_test_accs), 4))
    
    torch.save(train_process.clf.state_dict(), model_name)
    