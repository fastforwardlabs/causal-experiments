import random
import os
import shutil
import argparse
import pandas as pd

def create_data(args):
    if not os.path.exists(args['out_dir']):
        os.mkdir(args['out_dir'])
        
        train_df = pd.read_csv("../../../data/iWildCam/" + 'train.csv')
        print(train_df.head())
        classes = """empty, 0;deer, 1;moose, 2;squirrel, 3;rodent, 4;small_mammal, 5;elk, 6;pronghorn_antelope, 7;rabbit, 8;bighorn_sheep, 9;fox, 10;coyote, 11;black_bear, 12;raccoon, 13;skunk, 14;wolf, 15;bobcat, 16;cat, 17;dog, 18;opossum, 19;bison, 20;mountain_goat, 21;mountain_lion, 22""".split(';')
        
        classes = {int(i.split(', ')[1]): i.split(', ')[0] for i in classes}
        print("training classes: ", classes)
        
        train_df['classes'] = train_df['category_id'].apply(lambda x: classes[x])
        print(train_df.head())
        print("dataframe shape: ", train_df.shape)
        
        #train_df = train_df.drop_duplicates(subset='file_name', keep="first")
        
        test_data_dir = os.path.join(args['out_dir'], 'test')
        os.mkdir(test_data_dir)
        for i in args['animal_list']:
            os.mkdir(os.path.join(test_data_dir, i))  
            subset_test_filenames = train_df.loc[(train_df['classes'] == i) & (train_df['location'].isin(args['test_locations']))]
            for test_filename in subset_test_filenames['file_name'].tolist():
                filename = os.path.join(args['in_dir'], test_filename)
                shutil.copy2(filename, os.path.join(test_data_dir, i))      

        
        for loc in args['train_locations']:
            train_data_dir = 'train' + '_' + str(loc)
            os.mkdir(os.path.join(args['out_dir'], train_data_dir))
            for i in args['animal_list']:
                os.mkdir(os.path.join(args['out_dir'], train_data_dir, i))
                subset_train_filenames = train_df.loc[(train_df['classes'].isin([i])) & (train_df['location'].isin([loc]))]
                for train_filename in subset_train_filenames['file_name'].tolist():
                    filename = os.path.join(args['in_dir'], train_filename)
                    shutil.copy2(filename, os.path.join(args['out_dir'], train_data_dir, i))
        
        train_obs=0        
        for j in args['train_locations']:
            train_data_dir = 'train' + '_' + str(j)
            for i in args['animal_list']:
                print("for ", j , " and ", i, " ", len(os.listdir(os.path.join(args['out_dir'], train_data_dir, i))), "files")
                train_obs += len(os.listdir(os.path.join(args['out_dir'], train_data_dir, i)))
                
        test_obs=0  
        for i in args['animal_list']:
            print("for ", i , " ", len(os.listdir(os.path.join(test_data_dir, i))), "files")
            test_obs += len(os.listdir(os.path.join(test_data_dir, i)))
            
        print(train_obs, " files created in train dataset.")
        print(test_obs, " files created in test dataset.")
    else:
        print("Warning: output dir {} already exists".format(args['out_dir']))        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating WildCam train and test sets ..')
    parser.add_argument('--in_dir', type=str, default="../../../data/iWildCam/train/")
    parser.add_argument('--out_dir', type=str, default="../../../data/wildcam_subset_sample/")
    parser.add_argument('--animal_list', type=str, nargs='+', default=['raccoon', 'coyote'])
    parser.add_argument('--train_locations', type=int, nargs='+', default=[43, 46]) #[43, 46, 88, 130]
    parser.add_argument('--test_locations', type=int, nargs='+', default=[130]) #115
    args = dict(vars(parser.parse_args()))
    create_data(args)