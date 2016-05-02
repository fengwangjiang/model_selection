#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.datasets import make_classification
from make_datasets import make_dataset_2
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
from config import (N_SAMPLES, N_FEATURES, N_INFORMATIVE)


def gen_data(n_samples=N_SAMPLES, n_features=N_FEATURES,
             n_informative=N_INFORMATIVE,
             class_sep=2, type=1,
             random_state=None):
    """Generate type 1 data, var0 == var1, diagonal"""
    X, y = make_dataset_2(
        n_samples_1=n_samples//2, n_samples_2=n_samples//2,
        n_features=n_features, n_informative=n_informative,
        class_sep=class_sep, scale=2.25, type=type, G=20, rho=0.25,
        shuffle=True, random_state=random_state)
    return X, y


def gen_data2(n_samples=N_SAMPLES, n_features=N_FEATURES,
              n_informative=N_INFORMATIVE,
              class_sep=2, type=2,
              random_state=None):
    """Generate type 2 data, var0 != var1, diagonal"""
    X, y = make_dataset_2(
        n_samples_1=n_samples//2, n_samples_2=n_samples//2,
        n_features=n_features, n_informative=n_informative,
        class_sep=class_sep, scale=2.25, type=type, G=20, rho=0.25,
        shuffle=True, random_state=random_state)
    return X, y


def gen_data3(n_samples=N_SAMPLES, n_features=N_FEATURES,
              n_informative=N_INFORMATIVE, class_sep=2,
              n_clusters_per_class=2, random_state=None):
    """Generate data with 2 clusters per class"""
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=0,
                               n_repeated=0, n_classes=2,
                               n_clusters_per_class=n_clusters_per_class,
                               class_sep=class_sep,
                               random_state=random_state)
    return X, y


def gen_xor(n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_informative=N_INFORMATIVE,
            class_sep=2, scale=2.25, type=1, shuffle=True,
            random_state=None):
    """Generate xor data"""
    generator = check_random_state(random_state)
    n_samples_1 = n_samples//2
    n_samples_2 = n_samples//2
    n_samples_1_per_cluster = n_samples_1//2
    n_samples_2_per_cluster = n_samples_2//2
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)
    D = n_features
    if type == 1:
        mu1 = np.zeros(D)
        mu1[:n_informative] = class_sep
        cov1 = np.eye(D)
        X1_c1 = generator.multivariate_normal(mu1, cov1,
                                              size=n_samples_1_per_cluster)
        mu1[:n_informative] = -class_sep
        X1_c2 = generator.multivariate_normal(mu1, cov1,
                                              size=n_samples_1_per_cluster)

        mu2 = np.zeros(D)
        mu1[:n_informative] = class_sep * (-1)**np.arange(n_informative)
        cov2 = np.eye(D)
        X2_c1 = generator.multivariate_normal(mu2, cov2,
                                              size=n_samples_2_per_cluster)
        mu2[:n_informative] = class_sep * (-1) * (-1)**np.arange(n_informative)
        X2_c2 = generator.multivariate_normal(mu2, cov2,
                                              size=n_samples_2_per_cluster)
    X = np.vstack((X1_c1, X1_c2, X2_c1, X2_c2))

    # assign class labels to y
    y[:n_samples_1] = 0
    y[-n_samples_2:] = 1
    if shuffle:
        # Randomly permute samples.
        X, y = util_shuffle(X, y, random_state=generator)
        # Randomly permute features
        #  indices = np.arange(n_features)
        #  generator.shuffle(indices)
        #  X[:, :] = X[:, indices]
    #  print("Getting out of make_dataset_2")
    return X, y
