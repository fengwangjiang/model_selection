#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import (f_classif, SelectKBest)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from bolstered_helpers import bolstered_blobs
from gen_data import gen_data
from config import (initialization, data_abs_path, fig_abs_path)


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


def strip_filename_suffix(f_name):
    """docstring for strip_filename_suffix"""
    f_name = os.path.basename(f_name)
    base_name = os.path.splitext(f_name)[0]
    return base_name


def base_name_generator(clf_name, n_samples, n_features, n_informative):
    """docstring for base_name_generator"""
    #  figname = "hist_clf_{clf}_n_{n}_d_{d}_d0_{d0}.pdf"
    #  figname = figname.format(clf=clf_name, n=n, d=d, d0=d0)
    f_name_dict = dict(clf=clf_name, n=np.int(n_samples),
                       d=n_features, d0=n_informative)
    base_name = "clf_{clf}_n_{n}_d_{d}_d0_{d0}".format(**f_name_dict)
    return base_name


def txt_fname_gen(base_name):
    """docstring for txt_fname_gen"""
    return base_name + ".txt"


def srs_figname_gen(base_name):
    """structural risks figure name generator"""
    return "srs_" + base_name + ".pdf"


def hist_figname_gen(base_name):
    """histogram figure name generator"""
    return "hist_" + base_name + ".pdf"


def box_figname_gen(base_name):
    """boxplot figure name generator"""
    return "box_" + base_name + ".pdf"


def fname2basename(f_name):
    """Strip suffix, and prefix, return basename"""
    f_name = strip_filename_suffix(f_name)
    _idx = f_name.index("_")
    basename = f_name[_idx+1:]
    return basename


def file_name_parser(f_name):
    """docstring for file_name_parser"""
    base_name = strip_filename_suffix(f_name)
    name_list = base_name.split('_')
    #  clf_LDA_n_50_d_20_d0_3.txt
    #  hist_clf_LDA_n_50_d_20_d0_3.pdf
    #  srs_clf_LDA_n_50_d_20_d0_3.pdf
    # clf_LDA_n_100_D_200_d0_15_d_10_error.tsv
    # ['clf', 'n', 'd', 'd0']
    # ['LDA', '50', '20', '3']
    k = name_list[0::2]
    v = name_list[1::2]
    v = [x if i == 0 else np.int(x) for i, x in enumerate(v)]
    # ['LDA', 50, 20, 3]
    name_dict = dict(zip(k, v))
    return name_dict


def test_file_name_parser():
    """docstring for test_file_name_parser"""
    f_name = "clf_LDA_n_50_d_20_d0_3.txt"
    name_dict = file_name_parser(f_name)
    print(name_dict)


def sr(X, y, clf_name, k, bolster=False):
    """Structural risk.

    :param X:
    :param y:
    :param clf_name:
    :param k:             number of features to select
    :param bolster:
    """
    clf = choose_clf(clf_name)
    n, d = X.shape
    ve = "Selected {k} features out of {d}.".format(k=k, d=d)
    if k > d:
        raise ValueError(ve)
    #  X_, y_ = bolstered_blobs(X, y, new_bolster=True)
    X_, y_ = bolstered_blobs(X, y, new_bolster=False)
    linear_clfs = ["LDA", "LSVM"]
    if clf_name in linear_clfs:
        vc = k
        selector = SelectKBest(f_classif, k=k).fit(X, y)
        Xr = selector.transform(X)
        X_r = selector.transform(X_)
        clf.fit(Xr, y)
        if bolster:
            err = 1 - clf.score(X_r, y_)
        else:
            err = 1 - clf.score(Xr, y)
    elif clf_name == "KNN":
        vc = n / k * 1 / (n ** (1 / 5))
        clf.n_neighbors = k
        clf.fit(X, y)
        if bolster:
            err = 1 - clf.score(X_, y_)
        else:
            err = 1 - clf.score(X, y)
    err_bar = np.mean(err)
    #  print(err_bar)
    #  vc_confidence = np.sqrt(32/n*(k*np.log(n+1)))
    #  risk = err_bar + vc_confidence
    #  risk = err_bar + 2 * k / n
    #  risk = err_bar + np.sqrt(np.log(k)/n)
    #  risk = err_bar + 2 * np.log(k) / n
    #  risk = err_bar + 1 * np.log(k) / n
    vc_confidence = 2 * np.log(vc) / n
    #  vc_confidence = np.sqrt(np.log(k)/n)
    #  refer to elements of statistical learning by Hastie ebook page 241.
    #  \sigma^2 = N/(N-d)*err_bar, NOT WORKING.
    #  vc_confidence = 2 * k / n * n / (n-k) * err_bar
    risk = err_bar + vc_confidence
    return (err_bar, vc_confidence, risk)


def test_sr():
    """Plot SRM, deprecated"""
    #  clf = LDA()
    clf_name = "LDA"
    n = 50
    d = 20
    d0 = 3
    X, y = gen_data(n_samples=n, n_features=d, n_informative=d0,
                    class_sep=1.5)
    srs = np.zeros(shape=(d, 3))
    srs_bolster = np.zeros(shape=(d, 3))
    for i in range(d):
        k = i + 1
        srs[i, :] = sr(X, y, clf_name, k, bolster=False)
        srs_bolster[i, :] = sr(X, y, clf_name, k, bolster=True)
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


def ms_srm(X, y, clf_name):
    """model selection using structural risk minimization"""
    X, y = check_X_y(X, y)
    n, d = X.shape
    linear_clfs = ["LDA", "LSVM"]
    if clf_name in linear_clfs:
        num_ks = np.int(d/2)
        ks = range(1, num_ks+1)
    elif clf_name == "KNN":
        ks = range(3, 9+1, 2)
        num_ks = len(ks)
    srs = np.zeros(shape=(num_ks, 3))
    srs_bolster = np.zeros(shape=(num_ks, 3))
    for i, k in enumerate(ks):
        k = i + 1
        srs[i, :] = sr(X, y, clf_name, k, bolster=False)
        srs_bolster[i, :] = sr(X, y, clf_name, k, bolster=True)
    min_idx = np.argmin(srs[:, 2], axis=0)
    min_idx_bolster = np.argmin(srs_bolster[:, 2], axis=0)
    return (ks[min_idx], ks[min_idx_bolster], srs, srs_bolster)


def test_ms_srm(n=50, d=3, d0=3, clf_name="KNN"):
    """plot risks for model selection using SRM"""
    X, y = gen_data(n_samples=n, n_features=d, n_informative=d0,
                    class_sep=1.5)
    idx, idx_bolster, srs, srs_bolster = ms_srm(X, y, clf_name)
    print("idx:", idx)
    print("idx_bolster:", idx_bolster)
    print("srs:", srs)
    print("srs_bolster:", srs_bolster)


def _plot_risks(srs, srs_bolster, suptitle, figname=None):
    """Plot training error, VC-confidence, and risk

    :param srs:
    :param srs_bolster:
    :param figname:
    """
    d = srs.shape[0]

    # VC dimensions start from 1, in linear case, it is feature size
    ks = np.arange(1, d+1)
    #  for KNN, the complexity increases when K decreases.
    if figname.find("KNN") != -1:
        ks = np.arange(2*d-1, 0, -2)
    x = ks
    min_idx = np.argmin(srs[:, 2], axis=0)
    min_risk = np.min(srs[:, 2])
    min_idx_bolster = np.argmin(srs_bolster[:, 2], axis=0)
    min_risk_bolster = np.min(srs_bolster[:, 2])

    min_idxs = (ks[min_idx_bolster], ks[min_idx])
    min_risks = (min_risk_bolster, min_risk)

    srs_tuple = (srs_bolster, srs)

    fig, axes = plt.subplots(2, 1)
    titles = ["structure risk bolstered", "structure risk"]
    # VC dimensions start from 1, in linear case, it is feature size
    #  x = np.arange(1, d+1)
    #  for KNN, the complexity increases when K decreases.
    #  if figname.find("KNN") != -1:
    #  x = np.arange(2*d-1, 0, -2)
    for i, ax in enumerate(axes):
        ax.plot(x, srs_tuple[i][:, 0], label="training error")
        ax.plot(x, srs_tuple[i][:, 1], label="vc-confidence")
        ax.plot(x, srs_tuple[i][:, 2], label="bound on error")
        ax.plot([min_idxs[i], min_idxs[i]], [0, min_risks[i]], 'm--',
                linewidth=3)
        #  ax.set_xlim(1, d/2)
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.set_title(titles[i])
        if i == 0:
            ax.legend(fontsize='xx-small')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('bottom')
    fig.suptitle(suptitle)
    fig.set_tight_layout(True)
    if figname is None:
        figname = "srs_n_50_d_20_d0_3_test.pdf"
    fig.savefig(figname)
    plt.close()


def plot_ms_srm(n=50, d=20, d0=3, clf_name="LDA", figname=None):
    """plot risks for model selection using SRM"""
    #  clf = choose_clf(clf_name)
    X, y = gen_data(n_samples=n, n_features=d, n_informative=d0,
                    class_sep=1.5)
    idx, idx_bolster, srs, srs_bolster = ms_srm(X, y, clf_name)
    if figname is None:
        base_name = base_name_generator(clf_name, n, d, d0)
        figname = srs_figname_gen(base_name)
        figname = fig_abs_path(figname)
        #  figname = "srs_clf_{clf}_n_{n}_d_{d}_d0_{d0}.pdf"
        #  figname = figname.format(clf=clf_name, n=n, d=d, d0=d0)
    #  title = r"clf: {clf}, $n$={n}, $d$={d}, $d_0$={d0}"
    #  title = title.format(clf=clf_name, n=n, d=d, d0=d0)
    _plot_risks(srs, srs_bolster, suptitle="", figname=figname)


def choose_clf(clf_name="LDA"):
    """Choose a classifier according to `clf_name`"""
    if clf_name == "LDA":
        clf = LDA()
    elif clf_name == "LSVM":
        clf = SVC(kernel='linear')
    elif clf_name == "KNN":
        clf = KNeighborsClassifier()
    else:
        raise ValueError("Expect clf_name to be LDA or LSVM"
                         ", but found %s" % str(clf_name))
    return clf


def loop_ms_srm(n=50, d=20, d0=3, clf_name="LDA", nloop=100, fname=None):
    """Compare model selections using SRM"""
    #  clf = choose_clf(clf_name)
    if fname is None:
        base_name = base_name_generator(clf_name, n, d, d0)
        fname = txt_fname_gen(base_name)
        #  fname = "clf_{clf}_n_{n}_d_{d}_d0_{d0}.txt"
        #  fname = fname.format(clf=clf_name, n=n, d=d, d0=d0)
    idx_list = np.zeros(nloop, dtype=np.int)
    idx_bolster_list = np.zeros(nloop, dtype=np.int)
    for i in range(nloop):
        X, y = gen_data(n_samples=n, n_features=d, n_informative=d0,
                        class_sep=1.5)
        idx, idx_bolster, srs, srs_bolster = ms_srm(X, y, clf_name)
        idx_list[i] = idx
        idx_bolster_list[i] = idx_bolster
        print("In loop %s" % i)
    idx_list.shape = (-1, 1)
    idx_bolster_list.shape = (-1, 1)
    data = np.hstack((idx_list, idx_bolster_list))
    np.savetxt(fname, data, fmt="%d", delimiter="\t", header="resub\tbresub")
    return data


def read_loop_ms_srm(fname=None, histplot=True, boxplot=True):
    """read the file from model selection using SRM,
    plot histogram and/or boxplot if specified."""
    if fname is None:
        fname = "clf_LDA_n_50_d_20_d0_3.txt"
    basename = strip_filename_suffix(fname)
    data = np.genfromtxt(fname, dtype=np.int)
    idx_list, idx_bolster_list = data.transpose()

    f_dict = file_name_parser(fname)
    clf_name, n, d, d0 = (f_dict["clf"], f_dict["n"],
                          f_dict["d"], f_dict["d0"])
    if histplot:

        # evaluate model selection by resub, and bolstered resub
        e = np.mean(np.abs(idx_list - d0))
        e_bolster = np.mean(np.abs(idx_bolster_list - d0))
        fig, ax = plt.subplots()
        bins = np.arange(0.5, 2*d0-0.5+1)
        n_list, bins, patches = ax.hist([idx_list, idx_bolster_list],
                                        bins=bins,
                                        label=["resub", "bolstered resub"],
                                        color=["red", "blue"])
        title = r"clf: {clf}, $n$={n}, $d$={d}, $d_0$={d0}"
        title = title.format(clf=clf_name, n=n, d=d, d0=d0)
        ax.set_title(title)
        ax.legend()
        text = "Mean deviation\nresub: {e}\nbolstered resub: {e_b}"
        text = text.format(e=e, e_b=e_bolster)
        ax.text(0.66, 0.66, text, transform=ax.transAxes)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        hist_fig_name = hist_figname_gen(basename)
        hist_fig_name = fig_abs_path(hist_fig_name)
        fig.savefig(hist_fig_name)
        plt.close()

    if boxplot:
        fig, ax = plt.subplots()
        ax.boxplot([idx_list, idx_bolster_list])
        ax.set_xticklabels(["resub", "bresub"])

        xmin, xmax = ax.get_xlim()
        ax.hlines(d0, xmin=xmin, xmax=xmax, color="lightgrey")

        title = r"clf: {clf}, $n$={n}, $d$={d}, $d_0$={d0}"
        title = title.format(clf=clf_name, n=n, d=d, d0=d0)
        ax.set_title(title)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        box_fig_name = box_figname_gen(basename)
        box_fig_name = fig_abs_path(box_fig_name)
        fig.savefig(box_fig_name)
        plt.close()
    return (idx_list, idx_bolster_list)


def cmp_ms_srm(n=50, d=20, d0=3, clf_name="LDA", nloop=100, figname=None):
    """Compare model selections using SRM, deprecated"""
    #  clf = choose_clf(clf_name)
    if figname is None:
        figname = "hist_clf_{clf}_n_{n}_d_{d}_d0_{d0}.pdf"
        figname = figname.format(clf=clf_name, n=n, d=d, d0=d0)
    idx_list = np.zeros(nloop, dtype=np.int)
    idx_bolster_list = np.zeros(nloop, dtype=np.int)
    for i in range(nloop):
        X, y = gen_data(n_samples=n, n_features=d, n_informative=d0,
                        class_sep=1.5)
        idx, idx_bolster, srs, srs_bolster = ms_srm(X, y, clf_name)
        idx_list[i] = idx
        idx_bolster_list[i] = idx_bolster
        print("In loop %s" % i)
    # evaluate model selection by resub, and bolstered resub
    e = np.mean(np.abs(idx_list - d0))
    e_bolster = np.mean(np.abs(idx_bolster_list - d0))

    fig, ax = plt.subplots()
    bins = np.arange(0.5, 5.5+1)
    n_list, bins, patches = ax.hist([idx_list, idx_bolster_list], bins=bins,
                                    label=["resub", "bolstered resub"],
                                    color=["red", "blue"])
    title = r"clf: {clf}, $n$={n}, $d$={d}, $d_0$={d0}"
    title = title.format(clf=clf_name, n=n, d=d, d0=d0)
    ax.set_title(title)
    ax.legend()
    text = "Mean deviation\nresub: {e}\nbolstered resub: {e_b}"
    text = text.format(e=e, e_b=e_bolster)
    ax.text(0.66, 0.66, text, transform=ax.transAxes)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    fig.savefig(figname)
    plt.close()
    return (e, e_bolster)


def runner(*args, **kwargs):
    """run model selection, make some figures"""
    f_dict = kwargs
    clf_name, n, d, d0 = (f_dict["clf_name"], f_dict["n"],
                          f_dict["d"], f_dict["d0"])
    base_name = base_name_generator(clf_name, n, d, d0)
    fname = txt_fname_gen(base_name)
    fname = data_abs_path(fname)
    if not os.path.exists(fname):
        loop_ms_srm(**kwargs)
    read_loop_ms_srm(fname=fname)


if __name__ == '__main__':
    initialization()
    #  test_sr()
    #  srs = test_sr()

    #  deprecated
    #  e, e_bolster = cmp_ms_srm(nloop=100)
    #  print("LDA: e = {e}\ne_bolster = {e_b}".format(e=e, e_b=e_bolster))
    #  e, e_bolster = cmp_ms_srm(clf_name="LSVM", nloop=100)
    #  print("LSVM: e = {e}\ne_bolster = {e_b}".format(e=e, e_b=e_bolster))

    #  test_ms_srm(clf_name="KNN")

    #  linear classifiers
    #  plot_ms_srm(clf_name="LDA")
    #  plot_ms_srm(clf_name="LSVM")
    #  plot_ms_srm(d0=4, clf_name="LDA")
    #  plot_ms_srm(d0=4, clf_name="LSVM")

    plot_ms_srm(clf_name="KNN")

    #  loop_ms_srm()
    #  loop_ms_srm(clf_name="LSVM")
    #  read_loop_ms_srm()
    #  read_loop_ms_srm(fname="clf_LSVM_n_50_d_20_d0_3.txt")

    #  runner(n=50, d=20, d0=3, clf_name="LDA", nloop=100, fname=None)
    #  runner(n=50, d=20, d0=3, clf_name="LSVM", nloop=100, fname=None)
    #  runner(n=50, d=20, d0=4, clf_name="LDA", nloop=100, fname=None)
    #  runner(n=50, d=20, d0=4, clf_name="LSVM", nloop=100, fname=None)
    pass
