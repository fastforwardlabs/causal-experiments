import random
import os
import shutil
import argparse
import pandas as pd

def create_data(args):
    if not os.path.exists(args['out_dir']):
        os.mkdir(args['out_dir'])
        train_data_dir = os.path.join(args['out_dir'], 'train')
        val_data_dir = os.path.join(args['out_dir'], 'valid')
        os.mkdir(train_data_dir)
        os.mkdir(val_data_dir)
        
        train_df = pd.read_csv("../../../data/iWildCam/" + 'train.csv')
        print(train_df.head())
        classes = """empty, 0;deer, 1;moose, 2;squirrel, 3;rodent, 4;small_mammal, 5;elk, 6;pronghorn_antelope, 7;rabbit, 8;bighorn_sheep, 9;fox, 10;coyote, 11;black_bear, 12;raccoon, 13;skunk, 14;wolf, 15;bobcat, 16;cat, 17;dog, 18;opossum, 19;bison, 20;mountain_goat, 21;mountain_lion, 22""".split(';')
        
        classes = {int(i.split(', ')[1]): i.split(', ')[0] for i in classes}
        print("training classes: ", classes)
        
        train_df['classes'] = train_df['category_id'].apply(lambda x: classes[x])
        print(train_df.head())
        
        train_df = train_df.drop_duplicates(subset='file_name', keep="first")
        
        subset_train_filenames = train_df.loc[(train_df['classes'].isin(args['animal_list'])) & 
                                 (~train_df['location'].isin(args['val_locations']))]
        
        subset_valid_filenames = train_df.loc[(train_df['classes'].isin(args['animal_list'])) & 
                                 (train_df['location'].isin(args['val_locations']))]
        for i in args['animal_list']:
            os.mkdir(os.path.join(train_data_dir, i))
            os.mkdir(os.path.join(val_data_dir, i))        
        
        for train_filename in subset_train_filenames['file_name'].tolist():
            filename = os.path.join(args['in_dir'], train_filename)
            animal_name = subset_train_filenames.loc[subset_train_filenames['file_name'] == train_filename, 'classes'].item()
            shutil.copy2(filename, os.path.join(train_data_dir, animal_name))

        for valid_filename in subset_valid_filenames['file_name'].tolist():
            filename = os.path.join(args['in_dir'], valid_filename)
            animal_name = subset_valid_filenames.loc[subset_valid_filenames['file_name'] == valid_filename, 'classes'].item()
            shutil.copy2(filename, os.path.join(val_data_dir, animal_name))      
        
        train_obs=0
        val_obs=0
        for i in args['animal_list']:
            train_obs += len(os.listdir(os.path.join(train_data_dir, i)))
            val_obs += len(os.listdir(os.path.join(val_data_dir, i)))
        print(train_obs, " files created in train dataset.")
        print(val_obs, " files created in val dataset.")
    else:
        print("Warning: output dir {} already exists".format(args['out_dir']))        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating WildCam train and validation sets ..')
    parser.add_argument('--in_dir', type=str, default="../../../data/iWildCam/train/")
    parser.add_argument('--out_dir', type=str, default="../../../data/wildcam_subset/")
    parser.add_argument('--animal_list', type=str, nargs='+', default=['raccoon', 'coyote'])
    parser.add_argument('--val_locations', type=int, nargs='+', default=[130, 115])
    args = dict(vars(parser.parse_args()))
    create_data(args)