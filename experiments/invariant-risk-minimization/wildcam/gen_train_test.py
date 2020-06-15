"""
Script to write wildcam denoised data filenames to json.
"""

import torch
import json
import os
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import get_dataset, get_handler

# config
OUTPUT_FILE = 'train_test_filenames.json'
DATASET_NAME = 'WILDCAM'
DATASET_PATH = '/datapool/wildcam/wildcam_subset_denoised'

envs, x_test, y_test = get_dataset(DATASET_NAME, DATASET_PATH, overwrite=False)
x_env_0, y_env_0 = envs[0]['images'], envs[0]['labels']
x_env_1, y_env_1 = envs[1]['images'], envs[1]['labels']

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

x_all = np.concatenate([x_env_0, x_env_1, x_test])
y_all = np.concatenate([x_env_0, y_env_1, y_test])

n_data = len(x_all)

handler = get_handler(DATASET_NAME)

loader = DataLoader(handler(x_all, y_all, transform=transform), shuffle=False)

output = [] 

for idx, (x, y, _) in enumerate(loader):
    path = x_all[idx].replace(DATASET_PATH, '')
    _, location, category, filename = path.split("/")
    if location == "train_43":
        location = 43
    elif location == "train_46":
        location = 46
    else:
        location = 130

    output.append({
        'category': category,
        'location': location,
        'filename': filename
    })
    if idx % 100 == 0:
        print('predicted {} / {} images'.format(idx, n_data))        
# persist
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f)
