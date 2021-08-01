# Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation
The official code of the paper "Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation".

## Abstract

> Anomaly Detection is a well-known task that has been studied for decades. Anomalous events occur relatively infrequently, yet they can have serious and dangerous consequences in domains such as intrusion detection in cybersecurity, credit card fraud, health care, insurance and industrial damage.
Test-time augmentation (TTA) involves aggregating the predictions of several synthetic versions of a given test sample. The TTA is done to obtain a smoothed prediction.
To the best of our knowledge, no studies exist that utilize TTA for tabular anomaly detection. We propose a TTA-based method to improve the performance of anomaly detection. We took the test instances' nearest neighbors and generated its augmentations using the centroids of a k-Means model based on the instances’ neighbors. Our advanced approach utilizes a Siamese Network to learn the appropriate distance metric used when retrieving a test instance’s neighbors. We show that for all eight datasets we evaluated, the anomaly detector that used our TTA approach improved AUC significantly. Moreover, the learned distance metric approach showed a better improvement than the nearest neighbors model with the Euclidean distance metric.


## Repository Files

- ├──`data/`
  - └──`[dataset]/`
    - ├──`[dataset]_features.npy`: The preprocessed features of `[dataset]`
    - ├──`[dataset]_labels.npy`: The labels of `[dataset]`
    - ├──`[dataset]_pairs_X.npy`: The features of the pairs used for training the Siamese network of `[dataset]`
    - └──`[dataset]_pairs_y.npy`: The labels of the pairs used for training the Siamese network of `[dataset]`
- ├──`src/`: The source code implementation of our approach 


## Datasets

These are the datasets we used in our experiments and described in the paper.

As noted, the datasets were taken from [ODDS](http://odds.cs.stonybrook.edu/).
|Dataset|#Samples|#Dim|Outliers|
|:---:|:---:|:---:|:---:|
|[Annthyroid](http://odds.cs.stonybrook.edu/annthyroid-dataset/)|7200|6|7.42 (%)|
|[Cardio](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/)|1831|21|9.6 (%)|
|[Mammo](http://odds.cs.stonybrook.edu/mammography-dataset/)|11183|6|2.32 (%)|
|[Satellite](http://odds.cs.stonybrook.edu/satellite-dataset/)|6435|36|32 (%)|
|[Seismic](http://odds.cs.stonybrook.edu/seismic-dataset/)|2584|11|6.5 (%)|
|[Thyroid](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/)|3772|6|2.5 (%)|
|[Vowels](http://odds.cs.stonybrook.edu/japanese-vowels-data/)|1456|12|3.4 (%)|
|[Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast)|1364|8|4.7 (%)|


## Dependencies

The required dependencies are specified in `environment.yml`.

For setting up the environment, use [Anaconda](https://www.anaconda.com/):
```bash
$ conda env create -f environment.yml
$ conda activate adtta
```
