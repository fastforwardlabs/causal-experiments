"""
Script to write wildcam predictions from trained models to json.
"""


import torch
import json
import os
from PIL import Image
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from models import get_net, resnet18_extractor
from dataset import get_dataset, get_handler

from lime import lime_image

import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)
# config

IRM_MODEL_PATH = 'models/wildcam_denoised_121_0.001_40_10000.0_IRM.pth'
ERM_MODEL_PATH = 'models/wildcam_denoised_121_0.001_0_0.0_ERM.pth'
OUTPUT_FILE = 'output.json'

DATASET_NAME = 'WILDCAM'
DATASET_PATH = '/datapool/wildcam/wildcam_subset_denoised'


# data
'''
'''

transform = transforms.Compose([
    transforms.Resize((256, 256)),
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

#x_all = np.concatenate([x_env_0, x_env_1, x_test])
#y_all = np.concatenate([x_env_0, y_env_1, y_test])
x_all = x_test
y_all = y_test

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

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 
        
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    clf.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(device)
    batch = batch.to(device)
    
    logits = clf(batch)
    probs = torch.cat((1-torch.sigmoid(logits), torch.sigmoid(logits)), 1)
    return probs.detach().cpu().numpy()
# predict

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

output = [] 

for idx, (x, y, _) in enumerate(loader):
    if idx < 1:
        path = x_all[idx].replace(DATASET_PATH, '')
        label = 'raccoon' if y_all[idx] else 'coyote'

        #logit_irm, prob_irm, pred_irm  = predictions(irm_model, x, y)
        #logit_erm, prob_erm, pred_erm  = predictions(erm_model, x, y)
        pil_image = np.array(pill_transf(get_image(x_all[idx])))
        print(f'PIL image Min: {pil_image.min()}, Max: {pil_image.max()}')

        tensor_image = np.array(x[0])
        print(f'tensor image normalized Min: {tensor_image.min()}, Max: {tensor_image.max()}')
       
        #convert to numpy
        img_numpy = np.array(x[0].permute(1, 2, 0))

        clf = irm_model
        probs_irm = batch_predict([pill_transf(get_image(x_all[idx]))])        
        prob_irm = probs_irm[0][1]
        pred_irm = 'raccoon' if (prob_irm >= 0.5) else 'coyote'
        # use probability of the predicted class
        if pred_irm == 'coyote':
            prob_irm = 1 - prob_irm

        irm_explainer = lime_image.LimeImageExplainer(feature_selection='highest_weights', verbose=False, random_state=123)
        irm_explanation = irm_explainer.explain_instance(np.array(pill_transf(get_image(x_all[idx]))), 
                                         batch_predict, # classification function
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=1000, # number of images that will be sent to classification function
                                         random_seed=123) 

        clf = erm_model
        probs_erm = batch_predict([pill_transf(get_image(x_all[idx]))])
        prob_erm = probs_erm[0][1]
        pred_erm = 'raccoon' if (prob_erm >= 0.5) else 'coyote'
        # use probability of the predicted class
        if pred_erm == 'coyote':
            prob_erm = 1 - prob_erm
        erm_explainer = lime_image.LimeImageExplainer(feature_selection='highest_weights', verbose=False, random_state=123)
        erm_explanation = erm_explainer.explain_instance(np.array(pill_transf(get_image(x_all[idx]))), 
                                         batch_predict, # classification function
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=1000, # number of images that will be sent to classification function
                                         random_seed=123) 

        output.append({
            'image_path': path,
            'label': label,
            'image': img_numpy, #transformed image
            'irm': {
                #'logit': logit_irm.item(),
                'prob':  prob_irm.item(),
                'prediction': pred_irm,
                'segments': irm_explanation.segments,
                'coefficients': list(irm_explanation.local_exp.values())[0],
                'lime_prob': irm_explanation.local_pred[0],
                'lime_prediction': irm_explanation.top_labels[0]
            },
            'erm': {
                #'logit': logit_erm.item(),
                'prob':  prob_erm.item(),
                'prediction': pred_erm,
                'segments': erm_explanation.segments.tolist(),
                'coefficients': list(erm_explanation.local_exp.values())[0],
                'lime_prob': erm_explanation.local_pred[0],
                'lime_prediction': erm_explanation.top_labels[0]
            }
        })

        if idx % 10 == 0:
            print('predicted {} / {} images'.format(idx, n_data))


# persist

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, default=myconverter)

