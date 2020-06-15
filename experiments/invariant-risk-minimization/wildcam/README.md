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

NOTE: We use test images as they are unlabeled.

### create_data.py

Re-creates data folder with the following structure with disjoint locations in the train and test sets. The locations in this case correspond to different environments.

```
/wildcam_subset_sample
    /train
        /<location 1>
            /coyote
                /*.jpg
            /raccoon
                /*.jpg
        /<location 2>
            /coyote
                /*.jpg
            /raccoon
                /*.jpg
    /test
        /coyote
            /*.jpg
        /raccoon
            /*.jpg
```

**Alternately, you could skip all the data steps and use data on elephant located at /datapool/wildcam**

```
$ ls /datapool/wildcam
iWildCam wildcam_subset_sample
```

## Environment setup

There is a conda environment file, which can be used to set up the right python 3.7 environment with:

```bash
conda env create -f env.yaml
```

Alternatively, set the environment up manually with:

```
conda create --name irm python=3.7 ipykernel
conda activate irm
nvcc --version
ls /usr/local/
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
nvcc --version
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge matplotlib mlflow
conda install nb_conda
```

For Lime explanations - recreate the environment with python 2.7

```
conda create --name irm2.7 python=2.7 ipykernel
conda activate irm2.7
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge matplotlib
conda install nb_conda
conda install -c conda-forge lime
conda install scikit-learn
conda install scikit-image
conda install -c conda-forge progressbar
```

Create 'models' sub-folder

```
mkdir ./models
```

## Programs

* dataset.py - WildCam data loading
* models.py - Resnet18 models for feature extraction and/or fine-tuning
* main.py - entry point - change run parameters and data paths from here
* train.py - trains with both IRM and ERM approaches
* /notebooks - exploratory code, mainly to try and test things out. 
* /models - to save model runs

## Results

* model_results.txt - initial model results
* model_results_irm.txt - documented IRM model iterations
* model_results_erm.txt - documented ERM model iterations

We are grateful to the IRM authors and Facebook Research for their open [implementation](https://github.com/facebookresearch/InvariantRiskMinimization) ([LICENSE](https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/LICENSE)) of Invariant Risk Minimization.
