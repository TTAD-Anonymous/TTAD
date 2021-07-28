import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score, auc, precision_recall_curve

from tqdm import tqdm

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE

from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import KMeans

# define compared algorithms from the Experiments section
evaluated_algorithms = [
    'WO_TTA_Baseline',
    'Gaussian_TTA_Baseline',
    'Euclidean_SMOTE_TTA',
    'Siamese_SMOTE_TTA',
    'Euclidean_BorderlineSMOTE_TTA',
    'Siamese_BorderlineSMOTE_TTA',
    'Euclidean_Kmeans_TTA',
    'Siamese_Kmeans_TTA'
]

def test(X, folded_test_datasets_list, trained_estimators_list, trained_siamese_network, euclidean_nn_model, siamese_nn_model, args):
    """
    Performing test phase on the test set with all of the compared algorithms described in the Experiments section

    Parameters
    ----------
    X: ndarray of shape (#num_samples, #features). The dataset's features
    folded_test_datasets_list: list. The test set of each split in the k-fold
    trained_estimators_list: list. The trained estimator of each split in hte k-fold
    trained_siamese_network. TF's Model. The trained siamese internal model. used to obtain embedding of each test instance
    euclidean_nn_model: trained Neareset Neighbors model with euclidean distance metric
    siamese_nn_model: trained Neareset Neighbors model with Siamese distance metric
    args: argparse args. The args given to the program
    """

    # define k-fold split's metrics dict
    algorithms_folds_metrics_dict = {key: [] for key in evaluated_algorithms}

    for split_index in range(args.n_folds):
        # test set-up
        test_ds = folded_test_datasets_list[split_index]
        trained_encoder, trained_decoder = trained_estimators_list[split_index][0], trained_estimators_list[split_index][1]
        test_step_func = test_step()
        
        print(f"--- Testing k-fold split index: {split_index+1} ---")
        # testing current k-fold split
        current_split_algorithms_metrics = test_loop(X, test_ds, trained_encoder, trained_decoder, euclidean_nn_model, siamese_nn_model, trained_siamese_network, test_step_func, args)

        # update the folds metrics dictionary
        for algorithm, metrics in current_split_algorithms_metrics.items():
            algorithms_folds_metrics_dict[algorithm].append(metrics)
    
    for algorithm, folds_metrics in algorithms_folds_metrics_dict.items():
        algorithms_folds_metrics_dict[algorithm] = np.array(folds_metrics)
    
    # presenting results
    print_test_results(algorithms_folds_metrics_dict, args)

def test_loop(X, test_ds, trained_encoder, trained_decoder, euclidean_nn_model, siamese_nn_model, trained_siamese_network, test_step_func, args):
    """
    Performing the test loop with every evaluated algorithm.

    Parameters
    ----------
    X: numpy ndarray of shape (#num_samples, #features). The dataset's features
    test_ds: TF's Dataset. The test set
    trained_encoder: TF's Model. The trained encoder from the anomaly detector model
    trained_decoder: TF's Model. The trained decoder from the anomaly detector model
    euclidean_nn_model. trained Nearest Neighbors model with euclidean distance metric
    siamese_nn_model. trained Nearest Neighbors mkodel with siamese distance metric
    trained_siamese_network: TF's Model. The trained siamese model used for calculating two samples' distance
    test_step_func: function. The function that is used for performing single test step
    args: argparse args. The args given to the program
    """

    # loss function - with reduction equals to `NONE` in order to get the loss of every test example
    loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    num_neighbors = args.num_neighbors
    num_augmentations = args.num_augmentations

    algorithms_test_loss = {key: [] for key in evaluated_algorithms}
    algorithms_metrics = {key: [] for key in evaluated_algorithms}

    test_labels = []
    tqdm_total_bar = test_ds.cardinality().numpy()
    for step, (x_batch_test, y_batch_test) in tqdm(enumerate(test_ds), total=tqdm_total_bar):
        reconstruction_loss = test_step_func(x_batch_test, trained_encoder, trained_decoder, loss_func).numpy()
        algorithms_test_loss['WO_TTA_Baseline'].append(reconstruction_loss)
        test_labels.append(y_batch_test.numpy())

        # calculate euclidean nn indices
        euclidean_nn_batch_neighbors_indices = euclidean_nn_model.kneighbors(X=x_batch_test.numpy(), n_neighbors=num_neighbors, return_distance=False)
        # calculate siamese nn indices
        test_batch_latent_features = trained_siamese_network(x_batch_test).numpy()
        siamese_nn_batch_neighbors_indices = siamese_nn_model.kneighbors(X=test_batch_latent_features, n_neighbors=num_neighbors, return_distance=False)

        euclidean_nn_batch_neighbors_features = X[euclidean_nn_batch_neighbors_indices]
        siamese_nn_batch_neighbors_features = X[siamese_nn_batch_neighbors_indices]

        algorithms_tta_samples_dict = {
            'Gaussian_TTA_Baseline': generate_random_noise_tta_samples(x_batch_test.numpy(), num_augmentations=num_augmentations),
            'Euclidean_SMOTE_TTA': generate_oversampling_tta_samples(euclidean_nn_batch_neighbors_features, oversampling_method=SMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations),
            'Siamese_SMOTE_TTA': generate_oversampling_tta_samples(siamese_nn_batch_neighbors_features, oversampling_method=SMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations),
            'Euclidean_BorderlineSMOTE_TTA': generate_oversampling_tta_samples(euclidean_nn_batch_neighbors_features, oversampling_method=BorderlineSMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations),
            'Siamese_BorderlineSMOTE_TTA': generate_oversampling_tta_samples(siamese_nn_batch_neighbors_features, oversampling_method=BorderlineSMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations),
            'Euclidean_Kmeans_TTA': generate_kmeans_tta_samples(euclidean_nn_batch_neighbors_features, args.with_cuml, num_augmentations=num_augmentations),
            'Siamese_Kmeans_TTA': generate_kmeans_tta_samples(siamese_nn_batch_neighbors_features, args.with_cuml, num_augmentations=num_augmentations)
        }
        # making prediction (with the anomaly detection estimator) for every tta sample
        algorithms_tta_reconstruction_dict = {algorithm:test_step_func(tta_samples, trained_encoder, trained_decoder, loss_func).numpy() for algorithm, tta_samples  in algorithms_tta_samples_dict.items()}

        # merging given test sample's prediction with its tta predictions
        for algorithm, tta_reconstruction in algorithms_tta_reconstruction_dict.items():
            # combine original test samples' reconstruction with the kmeans regular-NN TTA samples' reconstruction
            for primary_loss, tta_loss in list(zip(reconstruction_loss, tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                algorithms_test_loss[algorithm].append(np.mean(combined_tta_loss))
    
    # flatten w/o tta baseline test loss and the test_labels vectors
    algorithms_test_loss['WO_TTA_Baseline'] = np.concatenate(algorithms_test_loss['WO_TTA_Baseline'], axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    y_true = np.asarray(test_labels).astype(int)

    # calculating AUC
    for algorithm in algorithms_metrics.keys():
        algorithms_metrics[algorithm].append(roc_auc_score(y_true, algorithms_test_loss[algorithm]))

    return algorithms_metrics

def test_step():
    # @tf.function
    def test_one_step(inputs, encoder, decoder, loss_func):
        latent_var = encoder(inputs)
        reconstructed = decoder(latent_var)
        reconstruction_loss = loss_func(inputs, reconstructed)

        return reconstruction_loss
    return test_one_step

def generate_random_noise_tta_samples(x_batch_test, num_augmentations):
    """
    Generating TTA with random Gaussian noise

    Parameters
    ----------
    x_batch_test: ndarray of shape (batch_size, #features). The features of each test sample in the batch
    num_augmentations: int. The nubmer of augmentations to produce 
    """

    size = x_batch_test.shape
    # scale = 0.2
    random_noise = np.random.normal(size=size)
    # adding the noise to the original batch test samples. expanding the middle dim of x_batch_test to make it (batch_size, 1, dataset_features_dim)
    gaussian_tta_samples = np.expand_dims(x_batch_test, axis=1) + random_noise

    return gaussian_tta_samples

def generate_kmeans_tta_samples(batch_neighbors_features, with_cuML, num_augmentations):
    """
    Generating TTA with trained k-means

    Parameters
    ----------
    batch_neighbors_features: numpy ndarray of shape (batch_size, num_neighbors, #features). The features of each neighbor of each test sample that is in the batch
    with_cuML: bool. If True, then using cuML's k-Means model otherwise using scikit-learn's k-Means model
    num_augmentations: int. The number of augmentations to produce
    """

    batch_tta_samples = []
    for neighbors_features in batch_neighbors_features:
        if with_cuML:
            kmeans_model = cuKMeans(n_clusters=num_augmentations, random_state=1234)
        else:
            kmeans_model = cuKMeans(n_clusters=num_augmentations, random_state=1234)
        neighbors_features = neighbors_features.astype(np.float32)
        kmeans_model.fit(X=neighbors_features)
        tta_samples = kmeans_model.cluster_centers_
        # appending to the batch tta samples
        batch_tta_samples.append(tta_samples)
    
    return np.array(batch_tta_samples)

def generate_oversampling_tta_samples(oversampling_batch_neighbors_features, num_neighbors, num_augmentations, oversampling_method):
    """
    Generating TTA with oversampling method (either SMOTE or Borderline-SMOTE)

    Parameters
    ----------
    oversampling_batch_neighbors_features: numpy ndarray of shape (batch_size, num_neighbors, #features). The features of each neighbor of each test sample that is in the batch
    num_neighbors: int. The number of neighbor each test sample in the batch has
    num_augmentations: int. The number of augmentations to produce
    oversampling_meethod: function. Either SMOTE or Borderline-SMOTE
    """

    batch_size, features_dim = oversampling_batch_neighbors_features.shape[0], oversampling_batch_neighbors_features.shape[-1]

    oversampling_batch_tta_samples = np.zeros((batch_size, num_augmentations, features_dim))
    for index_in_batch, original_neighbors_features in enumerate(oversampling_batch_neighbors_features):
        original_neighbors_labels = np.zeros((original_neighbors_features.shape[0],))
        
        # create fake samples for the imblearn dataset
        fake_neighbors_features = np.zeros((num_neighbors + num_augmentations, features_dim))
        fake_neighbors_labels = np.ones((fake_neighbors_features.shape[0],))

        # create the imblearn dataset
        imblearn_features = np.concatenate([original_neighbors_features, fake_neighbors_features])
        imblearn_labels = np.concatenate([original_neighbors_labels, fake_neighbors_labels])

        oversampling_obj = oversampling_method(k_neighbors=num_neighbors-1, random_state=42)
        X_res, y_res = oversampling_obj.fit_resample(imblearn_features, imblearn_labels)

        current_augmentations = X_res[-num_augmentations:]
        oversampling_batch_tta_samples[index_in_batch] = current_augmentations
    
    return oversampling_batch_tta_samples

def print_test_results(algorithms_folds_metrics, args):
    """
    Printing the results metrics of all of the evaluated algorithms

    Parameters
    ----------
    algorithms_folds_metrics: dictionaty. A dictionary that holds the results metrics for every algorithm in every k-fold split
    args: argparse args. The args given to the program
    """
    
    print(f"--- Reults on {args.n_folds}-fold cross-validation with {args.num_neighbors} neighbors and {args.num_augmentations} augmentations ---")
    for algorithm, folds_metrics in algorithms_folds_metrics.items():
        algorithm_name = algorithm.replace("_", " ")
        print("*"*100)
        print(f"--- {algorithm_name} ---")
        print_list = []
        for i in range(len(folds_metrics.mean(axis=0))):
            print_list.append(folds_metrics.mean(axis=0)[i])
            print_list.append(folds_metrics.std(axis=0)[i])
        print("AUC : {:0.3f}+-{:0.3f}".format(*print_list))
    
    print("*"*100)