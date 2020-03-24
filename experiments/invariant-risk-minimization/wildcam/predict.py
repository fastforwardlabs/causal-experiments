"""
Script to write wildcam predictions from trained models to json.
"""


import torch
import json
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from models import get_net, resnet18_extractor
from dataset import get_dataset, get_handler


# config

IRM_MODEL_PATH = 'models/wildcam_1501_0.001_40_10000.0_IRM.pth'
ERM_MODEL_PATH = 'models/wildcam_1501_0.001_0_0.0_ERM.pth'
OUTPUT_FILE = 'output.json'

DATASET_NAME = 'WILDCAM'
DATASET_PATH = '/datapool/wildcam/wildcam_subset_sample'


# data

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

envs, x_test, y_test = get_dataset(DATASET_NAME, DATASET_PATH, overwrite=False)
x_env_0, y_env_0 = envs[0]['images'], envs[0]['labels']
x_env_1, y_env_1 = envs[1]['images'], envs[1]['labels']

x_all = np.concatenate([x_env_0, x_env_1, x_test])
y_all = np.concatenate([x_env_0, y_env_1, y_test])

n_data = len(x_all)

handler = get_handler(DATASET_NAME)

loader = DataLoader(handler(x_all, y_all, transform=transform), shuffle=False)


# models

irm_model = resnet18_extractor()
irm_model.load_state_dict(torch.load(IRM_MODEL_PATH))
irm_model.eval()

erm_model = resnet18_extractor()
erm_model.load_state_dict(torch.load(ERM_MODEL_PATH))
erm_model.eval()


def predictions(model, x, y):
    logit = model(x)
    prob = torch.sigmoid(logit)
    pred = 'raccoon' if (prob >= 0.5) else 'coyote'
   
    # use probability of the predicted class
    if pred == 'coyote':
        prob = 1 - prob

    return logit.item(), prob.item(), pred


# predict

output = [] 

for idx, (x, y, _) in enumerate(loader):
    path = x_all[idx].replace(DATASET_PATH, '')
    label = 'raccoon' if y_all[idx] else 'coyote'

    logit_irm, prob_irm, pred_irm  = predictions(irm_model, x, y)
    logit_erm, prob_erm, pred_erm  = predictions(erm_model, x, y)

    output.append({
        'image_path': path,
        'label': label,
        'irm': {
            'logit': logit_irm,
            'prob':  prob_irm,
            'prediction': pred_irm
        },
        'erm': {
            'logit': logit_erm,
            'prob':  prob_erm,
            'prediction': pred_erm
        }
    })

    if idx % 100 == 0:
        print('predicted {} / {} images'.format(idx, n_data))


# persist

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f)

