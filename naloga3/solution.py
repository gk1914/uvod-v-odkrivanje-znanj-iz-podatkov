import numpy as np
from unidecode import unidecode
import re
import os
from collections import Counter
from collections import OrderedDict
import matplotlib.pyplot as plt


def kmers(s, k=3):
    """Generates k-mers for an input string."""
    for i in range(len(s)-k+1):
        yield s[i:i+k]


def eigenvalue(M, v):
    Mv = M.dot(v)
    return v.dot(Mv)


def prepare_data_matrix():
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf measure.
    """
    # create matrix X and list of languages
    X = []
    #languages = ["slv", "src1", "src3", "rus", "ukr", "pql", "czc", "lit", "lat", "est", "chn", "hng", "grk", "frn",
    #             "itn", "rum", "spn", "por", "eng", "ger", "dut", "swd", "nrn", "fin"]
    languages = ["slv", "src1", "src3", "rus", "ukr", "pql", "czc", "slo", "lit", "lat", "est", "trk", "hng", "grk", "frn",
                 "itn", "rum", "spn", "por", "eng", "ger", "dut", "swd", "nrn", "fin", "dns"]
    folder_path = "ready/"
    data = {}

    for file in os.listdir(folder_path):
        if file[:-4] not in languages:
            continue
        name = os.path.splitext(os.path.basename(file))[0]
        text = " ".join([line.strip() for line in open(folder_path + file, "rt", encoding="utf8").readlines()])
        text = unidecode(text).lower()
        text = re.sub(r'[\d!@#$%^&*()\-=_+|:;\'",.<>?]', '', text)
        text = " ".join(text.split())  # remove multiple whitespaces which occur from newlines in raw text (and possibly from removing characters)
        data[name] = Counter(kmers(text, 3))

    # ----------------------------------  IDF ranking --------------------------------------------------
    # |     not necessary to calculate exact IDF score - same result if we only count number of        |
    # |      documents the term appears in and sort in decreasing order (instead of increasing)        |
    # --------------------------------------------------------------------------------------------------
    # data => all triplets of letters and their count for each language
    # N => count of all languages/documents
    # contained => number of languages/documents that a tripple appears in
    N = len(languages)
    all_triplets = set()
    for lang in data.values():
        all_triplets = all_triplets | set(lang.keys())

    contained = Counter()
    for tri in all_triplets:
        for lang in data.values():
            if tri in lang.keys():
                contained[tri] += 1

    # find top 100 terms
    contained_srt = sorted(dict(OrderedDict(contained)).items(), key=lambda x: x[1], reverse=True)
    top100 = [tri for tri, _ in contained_srt if ' ' not in tri][:100]
    print("top100", top100)

    # prepare data matrix X
    top100_freq_by_language = []
    for lang in data.values():
        freq = [lang[tri] for tri in top100]
        top100_freq_by_language.append(tuple(freq))
    X = np.array(top100_freq_by_language)

    return X, languages


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """
    X = X - np.mean(X, axis=0)
    A = np.cov(X, rowvar=False)
    N, d = A.shape

    vec = np.random.rand(d)
    vec = vec / np.linalg.norm(vec)
    e = eigenvalue(A, vec)

    while True:
        Ar = A.dot(vec)
        v_new = Ar / np.linalg.norm(Ar)

        e_new = eigenvalue(A, v_new)
        if np.abs(e - e_new) < 0.00001:
            break

        vec = v_new
        e = e_new

    return vec, e


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    first, e1 = power_iteration(X)
    proj = X.dot(first)
    mat_new = np.outer(proj, first)
    X_new = X - mat_new
    second, e2 = power_iteration(X_new)
    return np.stack((first, second)), np.array([e1, e2])


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    X = X - np.mean(X, axis=0)
    return X.dot(vecs.T)


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    var_total = total_variance(X)
    var_explained = sum(eigenvalues)
    return var_explained / var_total


if __name__ == "__main__":

    # prepare the data matrix
    X, languages = prepare_data_matrix()

    # PCA
    eigenvectors, eigenvalues = power_iteration_two_components(X)
    transformed = project_to_eigenvectors(X, eigenvectors)
    explained_var = explained_variance_ratio(X, eigenvectors, eigenvalues)

    # plotting
    plt.title("Explained variance: " + str(explained_var))
    plt.scatter(transformed[:,0], transformed[:,1])
    for lang, point in zip(languages, transformed):
        plt.text(point[0], point[1], lang)
    plt.show()
