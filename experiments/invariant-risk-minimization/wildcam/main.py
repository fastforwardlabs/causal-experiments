import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import random
import mlflow

from dataset import get_dataset, get_handler
from models import get_net
from train import Train
from torchvision import transforms
from datetime import datetime

seed = 123
dataset_name = 'WILDCAM'
dataset_path = '/datapool/wildcam/wildcam_subset_sample'

args_pool = {
    'WILDCAM': {
        'n_restarts': 3,
        'steps': 5,
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
            'penalty_anneal_iters': 40, # make this 0 for ERM
            'penalty_weight': 10000.0 # make this 0 for ERM
        }
    }
}

args = args_pool[dataset_name]
model_name = args['model_path'] + "wildcam_" + str(args['steps']) + "_" + str(args['optimizer_args']['lr']) + "_" + str(args['optimizer_args']['penalty_anneal_iters']) + "_" + str(args['optimizer_args']['penalty_weight']) + "_"

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# load dataset
envs, x_test, y_test = get_dataset(dataset_name, dataset_path)
print("working with", len(envs), "training environments: ")
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



start = datetime.now()
with mlflow.start_run():
    
    mlflow.log_params({
        'steps': args['steps'],
        'lr': args['optimizer_args']['lr'],
        'penalty_anneal_iters': args['optimizer_args']['penalty_anneal_iters'],
        'penalty_weight': args['optimizer_args']['penalty_weight'],
        'l2_regularizer_weight': args['optimizer_args']['l2_regularizer_weight'],
        'train_batch_size': args['loader_tr_args']['batch_size']
    })
    
    print()
    if args['optimizer_args']['penalty_weight'] > 1.0:
        mlflow.log_param('method', 'IRM')
        model_name = model_name + "IRM.pth"
        print("========================================IRM========================================")
    else:
        mlflow.log_param('method', 'ERM')
        model_name = model_name + "ERM.pth"
        print("========================================ERM========================================")
    print(args)

    for restart in range(args['n_restarts']):  
        print("Restart", restart)
    
        with mlflow.start_run(nested=True):
            train_acc, test_acc = train_process.train()

    final_train_accs.append(train_acc)
    final_test_accs.append(test_acc)

    mlflow.log_metrics({
        'final_train_acc_mean': np.mean(final_train_accs),
        'final_train_acc_std': np.std(final_train_accs),
        'final_test_acc_mean': np.mean(final_test_accs),
        'final_test_acc_std': np.std(final_test_accs),
    })

    print('Final train acc (mean/std across restarts so far):')
    print(round(np.mean(final_train_accs), 4), round(np.std(final_train_accs), 4))
    print('Final test acc (mean/std across restarts so far):')
    print(round(np.mean(final_test_accs), 4), round(np.std(final_test_accs), 4))
    
    
end = datetime.now()

time_elapsed = end - start
print("time for training: ", time_elapsed.days, time_elapsed.min, "minutes and", time_elapsed.seconds, "seconds.")
torch.save(train_process.clf.state_dict(), model_name)
