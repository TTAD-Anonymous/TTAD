# ADTTA
Implementation of Unsupervised Diverse Test-Time Augmentation in Anomaly Detection

## Abstract

> Anomaly Detection is a well-known task that has been studied for decades. Anomalous events occur relatively infrequently, yet they can have serious and dangerous consequences in domains such as intrusion detection in cybersecurity, credit card fraud, health care, insurance and industrial damage.
Test-time augmentation (TTA) involves aggregating the predictions of several synthetic versions of a given test sample. The TTA is done to obtain a smoothed prediction.
To the best of our knowledge, no studies exist that utilize TTA for tabular anomaly detection. We propose a TTA-based method to improve the performance of anomaly detection. We took the test instances' nearest neighbors and generated its augmentations using the centroids of a k-Means model based on the instances’ neighbors. Our advanced approach utilizes a Siamese Network to learn the appropriate distance metric used when retrieving a test instance’s neighbors. We show that for all eight datasets we evaluated, the anomaly detector that used our TTA approach improved AUC significantly. Moreover, the learned distance metric approach showed a better improvement than the nearest neighbors model with the Euclidean distance metric.

## Dependencies

The required dependencies are specified in `environment.yml`.

For setting up the environment, use [Anaconda](https://www.anaconda.com/):
```bash
$ conda env create -f environment.yml
$ conda activate adtta
```
