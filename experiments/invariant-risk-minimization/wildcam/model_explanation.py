import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os, json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
from models import get_net
from lime import lime_image
from skimage.segmentation import mark_boundaries

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

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    #probs = torch.sigmoid(logits)
    # if you don't pass 2 probs, LIME always classifies all examples in the coyote category
    probs = torch.cat((1-torch.sigmoid(logits), torch.sigmoid(logits)), 1)
    return probs.detach().cpu().numpy()

def generate_explanations(images, outfile, num_samples, num_features, seed=123):
    img = get_image(images[0])
    test_pred = batch_predict([pill_transf(img), pill_transf(img)])
    print("test prediction logic", test_pred)
    fig = plt.figure(constrained_layout=True, figsize=(5, 20))
    spec = gridspec.GridSpec(ncols=3, nrows=len(images), figure=fig)
    i = 0
    for img in images:
        img = get_image(img)
        explainer = lime_image.LimeImageExplainer(feature_selection='highest_weights', verbose=True, random_state=123)
    
        explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                         batch_predict, # classification function
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=num_samples, # number of images that will be sent to classification function
                                         random_seed=seed) 
        print("label: ", explanation.top_labels[0])
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                            positive_only=True, negative_only=False, 
                                            num_features=num_features[i], hide_rest=True)
        img_boundry1 = mark_boundaries(temp/255.0, mask)
        
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                            positive_only=False, negative_only=True,
                                            num_features=num_features[i], hide_rest=True)
        img_boundry2 = mark_boundaries(temp/255.0, mask)
        
        f_ax1 = fig.add_subplot(spec[i, 0], xticks=[], yticks=[])
        f_ax2 = fig.add_subplot(spec[i, 1], xticks=[], yticks=[])
        f_ax3 = fig.add_subplot(spec[i, 2], xticks=[], yticks=[])
        f_ax1.imshow(img)
        f_ax2.imshow(img_boundry1)
        f_ax3.imshow(img_boundry2)
        i += 1
    plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0) # To save figure
    plt.show() # To show figure


if __name__ == "__main__":
    #model_filename="./models/wildcam_1501_0.001_40_10000.0_IRM.pth"
    #model_filename="./models/wildcam_1501_0.001_0_0.0_ERM.pth"
    #model_filename="./models/wildcam_denoised_121_0.001_40_10000.0_IRM.pth"
    model_filename="./models/wildcam_denoised_121_0.001_0_0.0_ERM.pth"
    net = get_net("WILDCAM")
    model = net(n_classes=2)
    print("loading model")
    model.load_state_dict(torch.load(model_filename, map_location="cpu"))
    model.to("cpu")

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()
    '''
    generate_explanations(images = 
                          [
                              #'../../../data/wildcam_subset_denoised/test/coyote/5903ccce-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/59373454-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/58adc310-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/58c7efed-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/59279c0b-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/5903cc2e-23d2-11e8-a6a3-ec086b02610b.jpg'
                              '../../../data/wildcam_subset_denoised/test/coyote/5865e36a-23d2-11e8-a6a3-ec086b02610b.jpg'
                          ], 
                          outfile='./figures/IRM_denoised_coyote_explanation.png', 
                          num_samples=1000, num_features=[10], seed=123)
    '''
    '''
    generate_explanations(images = 
                          [
                              #'../../../data/wildcam_subset_denoised/test/coyote/5903ccce-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/59373454-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/58adc310-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/58c7efed-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/59279c0b-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/coyote/5903cc2e-23d2-11e8-a6a3-ec086b02610b.jpg'
                              '../../../data/wildcam_subset_denoised/test/coyote/5865e36a-23d2-11e8-a6a3-ec086b02610b.jpg'
                          ], 
                          outfile='./figures/ERM_denoised_coyote_explanation.png', 
                          num_samples=1000, num_features=[10], seed=123)
    '''
    
    '''
    generate_explanations(images = 
                          [
                              #'../../../data/wildcam_subset_denoised/test/raccoon/593a4e8a-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/5879d289-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/58629252-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/591fd104-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/58a8a170-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/raccoon/5892b697-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/raccoon/58e40d0c-23d2-11e8-a6a3-ec086b02610b.jpg'
                              '../../../data/wildcam_subset_denoised/test/raccoon/58e2820f-23d2-11e8-a6a3-ec086b02610b.jpg'
                          ], 
                          outfile='./figures/IRM_denoised_raccoon_explanation.png', 
                          num_samples=1000, num_features=[20], seed=123)
    
    '''
    '''
    generate_explanations(images = 
                          [
                              #'../../../data/wildcam_subset_denoised/test/raccoon/593a4e8a-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/5879d289-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/58629252-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/591fd104-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/58a8a170-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/5892b697-23d2-11e8-a6a3-ec086b02610b.jpg',
                              #'../../../data/wildcam_subset_denoised/test/raccoon/58af7610-23d2-11e8-a6a3-ec086b02610b.jpg'
                              #'../../../data/wildcam_subset_denoised/test/raccoon/58e40d0c-23d2-11e8-a6a3-ec086b02610b.jpg'
                              '../../../data/wildcam_subset_denoised/test/raccoon/58732ea2-23d2-11e8-a6a3-ec086b02610b.jpg'
                          ], 
                          outfile='./figures/ERM_denoised_raccoon_explanation.png', 
                          num_samples=1000, num_features=[20], seed=123)
    
    '''
    '''
    generate_explanations(images = 
                          [
                              '../../../data/wildcam_subset_denoised/test/coyote/5903ccce-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/59373454-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/58c7efed-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/58adc310-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/59279c0b-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/5903cc2e-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/5865e36a-23d2-11e8-a6a3-ec086b02610b.jpg',
                              
                              '../../../data/wildcam_subset_denoised/test/raccoon/593a4e8a-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/5879d289-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/58629252-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/591fd104-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/58a8a170-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/58e40d0c-23d2-11e8-a6a3-ec086b02610b.jpg'
                           ], 
                          outfile='./figures/IRM_denoised_results.png', 
                          num_samples=1000, num_features=[10, 10, 10, 10, 10, 5, 10, 20, 10, 5, 20, 10, 20], seed=123)    
    '''
    
    generate_explanations(images = 
                          [
                              '../../../data/wildcam_subset_denoised/test/coyote/5903ccce-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/59373454-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/58c7efed-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/58adc310-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/59279c0b-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/5903cc2e-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/coyote/5865e36a-23d2-11e8-a6a3-ec086b02610b.jpg',
                              
                              '../../../data/wildcam_subset_denoised/test/raccoon/593a4e8a-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/5879d289-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/58629252-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/591fd104-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/58a8a170-23d2-11e8-a6a3-ec086b02610b.jpg',
                              '../../../data/wildcam_subset_denoised/test/raccoon/58e40d0c-23d2-11e8-a6a3-ec086b02610b.jpg'
                          ], 
                          outfile='./figures/ERM_denoised_results.png', 
                          num_samples=1000, num_features=[10, 10, 10, 10, 10, 5, 10, 20, 10, 5, 20, 10, 20], seed=123)
    