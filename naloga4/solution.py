import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from math import log


def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return 1 / (1 + np.e**(-x.dot(theta)))


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    reg = lambda_*sum(t**2 for t in theta)
    return -1/len(y) * sum(yi*log(max(h(xi, theta), 1e-100)) + (1-yi)*log(max(1-h(xi, theta), 1e-100)) for xi, yi in zip(X, y)) + reg


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne 1D numpy array v velikosti vektorja theta.
    """
    return np.array([-1/len(y) * sum((yi - h(xi, theta))*xi[j] for xi, yi in zip(X, y)) + lambda_*t
                     for j, t in enumerate(theta)])


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    eps = 1e-3
    return np.array([(cost(*(theta+i, X, y, lambda_)) - cost(*(theta-i, X, y, lambda_))) / (2 * eps)
                     for i in np.identity(len(theta)) * eps]) - lambda_*theta


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(LogRegLearner(lambda_=0.0), X, y)
    ... dopolnite (naloga 3)
    """
    m, n = X.shape
    print(m, n)
    step = int(m / k)

    max_ca = 0
    max_i = -1
    for i in range(k):
        print(i)
        learn_set = np.concatenate((X[:i*step, :], X[(i+1)*step:, :]), axis=0)
        print("learn", learn_set)
        print(learn_set.shape)
        y_learn = np.concatenate((y[:i*step], y[(i+1)*step:]))
        print("y_learn", y_learn)
        print(y_learn.shape)
        test_set = X[i*step:(i+1)*step, :]
        print("test", test_set)
        print(test_set.shape)
        y_test = y[i*step:(i+1)*step]
        classifier = learner(learn_set, y_learn)
        predictions = [classifier(t) for t in test_set]
        ca = CA(y_test, predictions)
        print(i, "... CA =", ca)
        if ca > max_ca:
            max_ca = ca
            max_i = i
    print("best acc =", max_ca, "(on iter", max_i, ")")
    print()
    print("....... using model from iter", max_i)
    learner_test = LogRegLearner(lambda_=0.0)
    classifier_test = learner_test(X, y)
    predictions = [classifier_test(x) for x in X]
    ca = CA(y, predictions)
    print("CA =", ca)
    return predictions


def CA(real, predictions):
    pred_classes = [np.argmax(p) for p in predictions]
    correct = [int(r == p) for r, p in zip(real, pred_classes)]
    return sum(correct) / len(correct)


def AUC(real, predictions):
    print("real", real)
    print("predictions", predictions)
    pred_classes = [np.argmax(p) for p in predictions]
    all = len(real)
    P = sum(real)
    N = all - P
    print(all, " = (P)", P, " + (N)", N)
    TP = 0
    FP = 0
    for r, p in zip(real, pred_classes):
        if r and p:
            TP += 1
        if not r and p:
            FP += 1
    FPr = FP / N
    TPr = TP / P
    print(FPr)
    print(TPr)

    return 1/2 - FPr/2 + TPr/2


if __name__ == "__main__":
    # Primer uporabe

    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X, y) # dobimo model

    napoved = classifier(X[0])  # napoved za prvi primer
    print(napoved)
