## Data

Use Kaggle account to download data. The Camera Traps (or Wild Cams) dataset - [iWildCam 2019](https://github.com/visipedia/iwildcam_comp) 

```
conda install -c conda-forge kaggle
cd causal-experiments/data

chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c iwildcam-2019-fgvc6
unzip iwildcam-2019-fgvc6.zip -d ./iWildCam

unzip train_images.zip -d ./train
unzip test_images.zip -d ./test
```

### create_data.py

Re-creates data folder with the following structure with disjoint locations in the train and validation sets

```
/wildcam_subset
    /train
        /coyote
            /*.jpg
        /raccoon
            /*.jpg
    /valid
        /coyote
            /*.jpg
        /raccoon
            /*.jpg
```

### sample data for modeling

Helps reduce runtime

```
mkdir wildcam_subset_sample
cd wildcam_subset_sample/
mkdir train
cd train
mkdir coyote
mkdir raccoon
find /home/nisha/causal-experiments/data/wildcam_subset/train/coyote/ -maxdepth 1 -type f |head -1000|xargs cp -t "/home/nisha/causal-experiments/data/wildcam_subset_sample/train/coyote"
find /home/nisha/causal-experiments/data/wildcam_subset/train/raccoon/ -maxdepth 1 -type f |head -1000|xargs cp -t "/home/nisha/causal-experiments/data/wildcam_subset_sample/train/raccoon"
cp -r /home/nisha/causal-experiments/data/wildcam_subset/valid /home/nisha/causal-experiments/data/wildcam_subset_sample
```

## Environment setup

```
conda create --name irm python=3.7 ipykernel
conda activate irm
nvcc --version
ls /usr/local/
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
nvcc --version
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge matplotlib
conda install nb_conda
```

## Programs

* dataset.py
* models.py
* main.py - entry point
* train.py

* main_mnist.py - code from the IRM research paper

## To-do

In the original paper code, the penalty is added every 100 steps/ epochs. The current code starts adding penalty_weight after some "n" epochs for every epoch. Once added the modified loss penalty should stay as is for a few epochs before being updated.
