import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from bolstered_helpers import bolstered_blobs
from gen_data import gen_data, gen_data2, gen_data3, gen_xor
from model_selection import (txt_fname_gen, err)
from config import data_abs_path, fig_abs_path
#  from model_selection import strip_filename_suffix
from model_selection import file_name_parser
from model_selection import fname2basename


def cmp_clfs(n=40, d=2, d0=2, data_model=1):
    """compare different classifiers

    plot classifier boundaries"""
    # generate some data to play with
    #  synthetic data model:
    #  1: var0 == var1, diagonal
    #  2: var0 != var1, diagonal
    #  3: data with 2 clusters per class
    #  4: xor data
    if data_model == 1:
        X, y = gen_data(n, d, d0, class_sep=1.5)
    elif data_model == 2:
        X, y = gen_data2(n, d, d0, class_sep=1.5)
    elif data_model == 3:
        X, y = gen_data3(n, d, d0, class_sep=1.5)
    elif data_model == 4:
        X, y = gen_xor(n, d, d0, class_sep=1.5)

    X_, y_ = bolstered_blobs(X, y, new_bolster=False)
    #  X_, y_ = bolstered_blobs(X, y, new_bolster=True)
    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    lda = LDA().fit(X, y)
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    poly_svc2 = svm.SVC(kernel='poly', degree=2, C=C).fit(X, y)
    poly_svc4 = svm.SVC(kernel='poly', degree=4, C=C).fit(X, y)
    poly_svc5 = svm.SVC(kernel='poly', degree=5, C=C).fit(X, y)
    tree_clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    knn_3 = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    knn_5 = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    knn_7 = KNeighborsClassifier(n_neighbors=7).fit(X, y)
    knn_9 = KNeighborsClassifier(n_neighbors=9).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['LDA',
              'SVM with linear kernel',
              'SVM with polynomial (degree 2) kernel',
              'SVM with RBF kernel',
              'SVM with polynomial (degree 3) kernel',
              'SVM with polynomial (degree 4) kernel',
              'SVM with polynomial (degree 5) kernel',
              'Decision tree classifier',
              'KNN with 3 nearest neighbors',
              'KNN with 5 nearest neighbors',
              'KNN with 7 nearest neighbors',
              'KNN with 9 nearest neighbors']

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    for i, clf in enumerate((lda, svc, poly_svc2, rbf_svc, poly_svc, poly_svc4,
                            poly_svc5, tree_clf, knn_3, knn_5, knn_7, knn_9)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        ax = fig.add_subplot(4, 3, i + 1)
        #  plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.2)

        # Plot also the training points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.rainbow, alpha=0.5)
        # Plot also the bolstered points
        #  ax.scatter(X_[:, 0], X_[:, 1], s=5, c=y_, cmap=plt.cm.rainbow)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(titles[i], fontsize='xx-small')

        e = 1 - clf.score(X, y)
        e_b = 1 - clf.score(X_, y_)
        text = "resub: {e:.2f}\nbolstered resub: {e_b:.2f}"
        text = text.format(e=e, e_b=e_b)
        ax.text(0.5, -0.1, text, fontsize=5, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

    figname_dict = dict(n=n, d=d, d0=d0, model=data_model)
    figname = "cmp_clfs_n_{n}_d_{d}_d0_{d0}_model_{model}.pdf"
    figname = figname.format(**figname_dict)
    fig.savefig(figname)
    plt.close()


def run_cmp_clfs():
    """run different data models"""
    for dm in [1, 2, 3, 4]:
        cmp_clfs(data_model=dm)


def loop_errs(n=40, d=2, d0=2, nloop=100, data_model=1, fname=None):
    """Calculate errors for different classifiers

    generate data using `n`, `d`, `d0`, `data_model` parameters
    calculate resub and bresub for different classifiers
    run it for `nloop` times
    save errors to tsv file `fname`."""
    clf_names = ["LDA", "LSVM", "RBF", "POLY2", "POLY3", "POLY4", "POLY5",
                 "CART", "3NN", "5NN", "7NN", "9NN"]
    C = 1.0  # SVM regularization parameter
    clfs = [LDA(),
            svm.SVC(kernel='linear', C=C),
            svm.SVC(kernel='rbf', gamma=0.7, C=C),
            svm.SVC(kernel='poly', degree=2, C=C),
            svm.SVC(kernel='poly', degree=3, C=C),
            svm.SVC(kernel='poly', degree=4, C=C),
            svm.SVC(kernel='poly', degree=5, C=C),
            DecisionTreeClassifier(max_depth=2),
            KNeighborsClassifier(n_neighbors=3),
            KNeighborsClassifier(n_neighbors=5),
            KNeighborsClassifier(n_neighbors=7),
            KNeighborsClassifier(n_neighbors=9)]
    num_clf = len(clf_names)
    err_names = clf_names.copy()
    berr_names = ["b" + x for x in err_names]
    header = err_names + berr_names
    header = '\t'.join(header)

    if fname is None:
        f_name_dict = dict(n=n, d=d, d0=d0, t=data_model)
        base_name = "errs_n_{n}_d_{d}_d0_{d0}_model_{t}".format(**f_name_dict)
        fname = txt_fname_gen(base_name)
        fname = data_abs_path(fname)

    errs = np.zeros((nloop, 2*num_clf))

    for i in range(nloop):
        if data_model == 1:
            X, y = gen_data(n, d, d0, class_sep=1.5)
        elif data_model == 2:
            X, y = gen_data2(n, d, d0, class_sep=1.5)
        elif data_model == 3:
            X, y = gen_data3(n, d, d0, class_sep=1.5)
        elif data_model == 4:
            X, y = gen_xor(n, d, d0, class_sep=1.5)

        errors = errs[i]

        for j, clf in enumerate(clfs):
            resub = err(X, y, clf)
            bresub = err(X, y, clf, bolster=True)
            errors[[j, j+num_clf]] = [resub, bresub]
        print("In loop %s" % i)

    np.savetxt(fname, errs, fmt="%.3f", delimiter="\t", header=header)
    return errs


def run_loop_errs(nloop=100):
    """run loop_errs different data models"""
    for dm in [1, 2, 3, 4]:
        loop_errs(nloop=nloop, data_model=dm)


def read_loop_errs(fname=None):
    """Read errors from tsv file `fname`

    typically after running run_loop_errs, which save errors
    for different classifiers to a file;
    This function returns the errors in file `fname`."""

    if fname is None:
        fname = "errs_n_40_d_2_d0_2_model_1"
    basename = fname2basename(fname)
    f_dict = file_name_parser(basename)
    print(f_dict)
    #  model, n, d, d0 = (f_dict["model"], f_dict["n"],
    #  f_dict["d"], f_dict["d0"])

    #  fname = txt_fname_gen(fname)
    #  fname = data_abs_path(fname)

    errs = np.genfromtxt(fname)
    #  print("errs:\n{}".format(errs))
    #  errs_mean = np.mean(errs, axis=0)
    #  print("errs_mean:\n{}".format(errs_mean))
    return errs


def plot_mean_errs(errs, figname=None):
    """Plot the mean of errors for different classifiers"""
    errs_mean = np.mean(errs, axis=0)
    errs_std = np.std(errs, axis=0)
    clf_names = ["LDA", "LSVM", "RBF", "POLY2", "POLY3", "POLY4", "POLY5",
                 "CART", "3NN", "5NN", "7NN", "9NN"]
    num_clf = len(clf_names)
    x = np.arange(1, num_clf+1)
    y = np.arange(0.0, 0.51, 0.1)
    resub_mean = errs_mean[:num_clf]
    bresub_mean = errs_mean[num_clf:]
    resub_std = errs_std[:num_clf]
    bresub_std = errs_std[num_clf:]
    resub_mean_std_min_idx = np.argmin(resub_mean + resub_std)
    bresub_mean_std_min_idx = np.argmin(bresub_mean + bresub_std)
    resub_mean_std_min_idx = np.argmin(resub_mean)
    bresub_mean_std_min_idx = np.argmin(bresub_mean)
    #  resub = errs[:, :num_clf]
    #  bresub = errs[:, num_clf:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    #  boxprops = dict(color='red')
    #  boxprops_b = dict(color='blue')
    #  ax.boxplot(resub, boxprops=boxprops)
    #  ax.boxplot(bresub, boxprops=boxprops_b)
    ax.plot(x, resub_mean, label="resub", ls='--', color='r', marker='o',
            ms=5, fillstyle='none')
    ax.plot(x[resub_mean_std_min_idx], resub_mean[resub_mean_std_min_idx],
            ls='None', color='r', marker='o', ms=5)
    ax.plot(x, bresub_mean, label="bresub", ls='-', color='b', marker='^',
            ms=5, fillstyle='none')
    ax.plot(x[bresub_mean_std_min_idx], bresub_mean[bresub_mean_std_min_idx],
            ls='None', color='b', marker='^', ms=5)
    ax.set_xlim(0.5, num_clf + 0.5)
    #  ax.set_title("resub and bresub error")
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, fontsize='xx-small', rotation=30)
    ax.set_yticks(y)
    ax.set_yticklabels(y, fontsize='xx-small')
    ax.legend(fontsize='xx-small')
    ax.yaxis.grid(color='grey')
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    logical_x_range = num_clf    # Bark
    logical_y_range = 0.5    # dB
    physical_x_range = 8     # inch
    physical_y_range = 3  # inch
    ax.set_aspect(
        (physical_y_range/logical_y_range) /
        (physical_x_range/logical_x_range))
    if figname is None:
        figname = "errs_n_40_d_2_d0_2_model_1.pdf"
    fig.suptitle("resub and bresub error")
    fig.savefig(figname)
    plt.close()
    return errs_mean, errs_std


def run_plot_mean_errs():
    """run plot_mean_errs to plot mean errors."""
    n = 40
    d = 2
    d0 = 2
    data_models = [1, 2, 3, 4]
    for dm in data_models:
        f_name_dict = dict(n=n, d=d, d0=d0, t=dm)
        base_name = "errs_n_{n}_d_{d}_d0_{d0}_model_{t}".format(**f_name_dict)
        fname = txt_fname_gen(base_name)
        fname = data_abs_path(fname)

        figname = base_name + '.pdf'
        figname = fig_abs_path(figname)

        errs = read_loop_errs(fname=fname)
        errs_mean, errs_std = plot_mean_errs(errs, figname=figname)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("errs_mean:\n{}".format(errs_mean))
        print("errs_std:\n{}".format(errs_std))


if __name__ == '__main__':
    #  run_cmp_clfs()
    #  run_loop_errs(nloop=100)
    run_plot_mean_errs()
