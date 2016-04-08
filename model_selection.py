#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.feature_selection import (f_classif, SelectKBest)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from bolstered_helpers import bolstered_blobs
from gen_data import gen_data
import matplotlib.pyplot as plt


def _num_samples(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    uniques = np.unique([_num_samples(X) for X in arrays if X is not None])
    if len(uniques) > 1:
        raise ValueError("Found arrays with inconsistent numbers of samples: "
                         "%s" % str(uniques))


def check_X_y(X, y):
    """Input validation for standard estimators.
    Checks X and y for consistent length, enforces X 2d and y 1d.
    """
    check_consistent_length(X, y)

    return X, y


def sr(X, y, clf, k, bolster=False):
    """Structural risk.

    :param X:
    :param y:
    :param clf:
    :param k:             number of features to select
    :param bolster:
    """
    n, d = X.shape
    ve = "Selected {k} features out of {d}.".format(k=k, d=d)
    if k > d:
        raise ValueError(ve)
    #  X_, y_ = bolstered_blobs(X, y, new_bolster=True)
    X_, y_ = bolstered_blobs(X, y, new_bolster=False)
    selector = SelectKBest(f_classif, k=k).fit(X, y)
    Xr = selector.transform(X)
    X_r = selector.transform(X_)
    clf.fit(Xr, y)
    if bolster:
        err = 1 - clf.score(X_r, y_)
    else:
        err = 1 - clf.score(Xr, y)
    err_bar = np.mean(err)
    #  print(err_bar)
    #  vc_confidence = np.sqrt(32/n*(k*np.log(n+1)))
    #  risk = err_bar + vc_confidence
    #  risk = err_bar + 2 * k / n
    #  risk = err_bar + np.sqrt(np.log(k)/n)
    #  risk = err_bar + 2 * np.log(k) / n
    #  risk = err_bar + 1 * np.log(k) / n
    vc_confidence = 2 * np.log(k) / n
    #  vc_confidence = np.sqrt(np.log(k)/n)
    risk = err_bar + vc_confidence
    return (err_bar, vc_confidence, risk)


def ms_srm(X, y, clf):
    """model selection using structural risk minimization"""
    X, y = check_X_y(X, y)
    n, d = X.shape
    srs = np.zeros(shape=(d, 3))
    srs_bolster = np.zeros(shape=(d, 3))
    for i in range(d):
        k = i + 1
        srs[i, :] = sr(X, y, clf, k, bolster=False)
        srs_bolster[i, :] = sr(X, y, clf, k, bolster=True)
    min_idx = np.argmin(srs[:, 2], axis=0) + 1
    min_idx_bolster = np.argmin(srs_bolster[:, 2], axis=0) + 1
    return (min_idx, min_idx_bolster, srs, srs_bolster)


def test_ms_srm(clf_name="LDA"):
    """plot risks for model selection using SRM"""
    if clf_name == "LDA":
        clf = LDA()
    elif clf_name == "LSVM":
        clf = SVC(kernel='linear')
    else:
        raise ValueError("Expect clf_name to be LDA or LSVM"
                         ", but found %s" % str(clf_name))
    n = 50
    d = 20
    d0 = 3
    X, y = gen_data(n_samples=n, n_features=d, n_informative=d0,
                    class_sep=1.5)
    idx1, idx2, srs, srs_bolster = ms_srm(X, y, clf)
    figname = "srs_clf_{clf}_n_{n}_d_{d}_d0_{d0}.pdf"
    figname = figname.format(clf=clf_name, n=n, d=d, d0=d0)
    plot_risks(srs, srs_bolster, figname=figname)


def plot_risks(srs, srs_bolster, figname="srs_n_50_d_20_d0_3_better.pdf"):
    """Plot training error, VC-confidence, and risk

    :param srs:
    :param srs_bolster:
    :param figname:
    """
    d = srs.shape[0]
    min_idx = np.argmin(srs[:, 2], axis=0) + 1
    min_risk = np.min(srs[:, 2])
    min_idx_bolster = np.argmin(srs_bolster[:, 2], axis=0) + 1
    min_risk_bolster = np.min(srs_bolster[:, 2])

    min_idxs = (min_idx_bolster, min_idx)
    min_risks = (min_risk_bolster, min_risk)

    srs_tuple = (srs_bolster, srs)

    fig, axes = plt.subplots(2, 1)
    titles = ["structure risk bolstered", "structure risk"]
    # VC dimensions start from 1, in linear case, it is feature size
    x = np.arange(1, d+1)
    for i, ax in enumerate(axes):
        ax.plot(x, srs_tuple[i][:, 0], label="training error")
        ax.plot(x, srs_tuple[i][:, 1], label="vc-confidence")
        ax.plot(x, srs_tuple[i][:, 2], label="bound on error")
        ax.plot([min_idxs[i], min_idxs[i]], [0, min_risks[i]], 'k--')
        ax.set_xlim(1, d/2)
        ax.set_title(titles[i])
        if i == 0:
            ax.legend(fontsize='xx-small')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    fig.set_tight_layout(True)
    fig.savefig(figname)
    plt.close()


def test_sr():
    """Plot SRM, deprecated"""
    clf = LDA()
    n = 50
    d = 20
    d0 = 3
    X, y = gen_data(n_samples=n, n_features=d, n_informative=d0,
                    class_sep=1.5)
    srs = np.zeros(shape=(d, 3))
    srs_bolster = np.zeros(shape=(d, 3))
    for i in range(d):
        k = i + 1
        srs[i, :] = sr(X, y, clf, k, bolster=False)
        srs_bolster[i, :] = sr(X, y, clf, k, bolster=True)
    min_idx = np.argmin(srs[:, 2], axis=0) + 1
    min_risk = np.min(srs[:, 2])
    min_idx_bolster = np.argmin(srs_bolster[:, 2], axis=0) + 1
    min_risk_bolster = np.min(srs_bolster[:, 2])
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(np.arange(d)+1, srs_bolster[:, 0], label="training error")
    ax1.plot(np.arange(d)+1, srs_bolster[:, 1], label="vc-confidence")
    ax1.plot(np.arange(d)+1, srs_bolster[:, 2], label="bound on error")
    ax1.plot([min_idx_bolster, min_idx_bolster], [0, min_risk_bolster], 'k--')
    ax1.set_xlim(1, d/2)
    ax1.set_title("structure risk bolstered")
    ax1.legend(fontsize='xx-small')
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax2.plot(np.arange(d)+1, srs[:, 0], label="training error")
    ax2.plot(np.arange(d)+1, srs[:, 1], label="vc-confidence")
    ax2.plot(np.arange(d)+1, srs[:, 2], label="bound on error")
    ax2.plot([min_idx, min_idx], [0, min_risk], 'k--')
    ax2.set_xlim(1, d/2)
    ax2.set_title("structure risk")
    # Hide the right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    fname = "srs_n_{n}_d_{d}_d0_{d0}.pdf".format(n=n, d=d, d0=d0)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    #  return min_idx, min_idx_bolster


if __name__ == '__main__':
    #  test_sr()
    #  srs = test_sr()
    #  min_idx = np.argmin(srs, axis=0)
    #  ks = min_idx + 1
    #  msg = "\tbresub\tresub\nk\t{0}\t{1}\n".format(*ks)
    #  print(msg)
    test_ms_srm(clf_name="LDA")
    test_ms_srm(clf_name="LSVM")
