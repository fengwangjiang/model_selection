#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from make_datasets import make_dataset_2
from config import (N_SAMPLES, N_FEATURES, N_INFORMATIVE)


def gen_data(n_samples=N_SAMPLES, n_features=N_FEATURES,
             n_informative=N_INFORMATIVE,
             class_sep=2, type=1,
             random_state=None):
    """docstring for """
    X, y = make_dataset_2(
        n_samples_1=n_samples//2, n_samples_2=n_samples//2,
        n_features=n_features, n_informative=n_informative,
        class_sep=class_sep, scale=2.25, type=type, G=20, rho=0.25,
        shuffle=True, random_state=random_state)
    return X, y
