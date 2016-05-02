import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from bolstered_helpers import bolstered_blobs
from gen_data import gen_data, gen_data2, gen_data3, gen_xor


def cmp_clfs(n=40, d=2, d0=2, data_model=1):
    """compare different classifiers"""
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


if __name__ == '__main__':
    run_cmp_clfs()
