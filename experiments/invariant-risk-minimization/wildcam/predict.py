"""
Script to write wildcam predictions from trained models to json.
"""

import torch
import json
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchvision import transforms
from torch.utils.data import DataLoader

from models import get_net, resnet18_extractor
from dataset import get_dataset, get_handler

from lime import lime_image
from PIL import Image

from skimage.segmentation import mark_boundaries

# config

IRM_MODEL_PATH = 'models/wildcam_denoised_121_0.001_40_10000.0_IRM.pth'
ERM_MODEL_PATH = 'models/wildcam_denoised_121_0.001_0_0.0_ERM.pth'
OUTPUT_FILE = 'output_wo_explanations.json'

DATASET_NAME = 'WILDCAM'
DATASET_PATH = '/datapool/wildcam/wildcam_subset_denoised'

# data
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

x_all = np.concatenate([x_env_0, x_env_1, x_test])
y_all = np.concatenate([x_env_0, y_env_1, y_test])
#x_all = x_test
#y_all = y_test

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

def myconverter(obj):
    '''
    this is to make the objects serializable while writing to json
    '''
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()

output = [] 
num_samples = 10
#fig = plt.figure(constrained_layout=True, figsize=(5, 20))
#spec = gridspec.GridSpec(ncols=2, nrows=num_samples, figure=fig)
#i=0

for idx, (x, y, _) in enumerate(loader):
    #if idx < num_samples:        
    path = x_all[idx].replace(DATASET_PATH, '')
    label = 'raccoon' if y_all[idx] else 'coyote'

    # image that will be shown to the user, this is also needed by the LIME api
    pil_image = np.array(pill_transf(get_image(x_all[idx])))
        
    clf = irm_model
    probs_irm = batch_predict([pill_transf(get_image(x_all[idx]))])  
        
    prob_irm = probs_irm[0][1]
    pred_irm = 'raccoon' if (prob_irm >= 0.5) else 'coyote'
    # use probability of the predicted class
    if pred_irm == 'coyote':
        prob_irm = 1 - prob_irm
    '''
    irm_explainer = lime_image.LimeImageExplainer(feature_selection='highest_weights', verbose=False, random_state=123)
    irm_explanation = irm_explainer.explain_instance(pil_image, #this cannot be a tensor and has to be [H, W, C]
                                         batch_predict, # classification function
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=1000, # number of images that will be sent to classification function
                                         random_seed=123) 
    '''
    clf = erm_model
    probs_erm = batch_predict([pill_transf(get_image(x_all[idx]))])
    prob_erm = probs_erm[0][1]
    pred_erm = 'raccoon' if (prob_erm >= 0.5) else 'coyote'
    # use probability of the predicted class
    if pred_erm == 'coyote':
        prob_erm = 1 - prob_erm
    '''
    erm_explainer = lime_image.LimeImageExplainer(feature_selection='highest_weights', verbose=False, random_state=123)
    erm_explanation = erm_explainer.explain_instance(pil_image, 
                                         batch_predict, # classification function
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=1000, # number of images that will be sent to classification function
                                         random_seed=123) 
    '''
    '''
        irm_temp, irm_mask = irm_explanation.get_image_and_mask(irm_explanation.top_labels[0], 
                                            positive_only=True, negative_only=False, 
                                            num_features=5, hide_rest=False)
        irm_img_boundary = mark_boundaries(irm_temp/255.0, irm_mask)
        
        erm_temp, erm_mask = erm_explanation.get_image_and_mask(erm_explanation.top_labels[0], 
                                            positive_only=True, negative_only=False,
                                            num_features=5, hide_rest=False)
        erm_img_boundary = mark_boundaries(erm_temp/255.0, erm_mask)
        
        f_ax1 = fig.add_subplot(spec[i, 0], xticks=[], yticks=[])
        f_ax2 = fig.add_subplot(spec[i, 1], xticks=[], yticks=[])

        f_ax1.imshow(irm_img_boundary)
        f_ax2.imshow(erm_img_boundary)

        i += 1
    '''
    output.append({
        'image_path': path,
        'label': label,
        #'image': pil_image, # resized & cropped image
        'irm': {
            'prob':  prob_irm.item(),
            'prediction': pred_irm,
            #'segments': irm_explanation.segments,
            #'coefficients': list(irm_explanation.local_exp.values())[0],
            #'lime_prob': irm_explanation.local_pred[0]
        },
        'erm': {
            'prob':  prob_erm.item(),
            'prediction': pred_erm,
            #'segments': erm_explanation.segments.tolist(),
            #'coefficients': list(erm_explanation.local_exp.values())[0],
            #'lime_prob': erm_explanation.local_pred[0]
        }
    })

    if idx % 100 == 0:
        print('predicted {} / {} images'.format(idx, n_data))

#plt.savefig('./figures/predict.png', dpi=300, bbox_inches='tight', pad_inches=0) # To save figure
#plt.show() # To show figure

# persist
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, default=myconverter)