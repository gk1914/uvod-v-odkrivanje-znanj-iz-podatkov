import csv
from collections import defaultdict
import math
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(file_name):
    with open(file_name, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        data = list(reader)
        #print(data)
        return data


def write_to_file(file_name, recommendations):
    fo = open(file_name, "wt")
    for r in recommendations:
        fo.write(str(r) + "\n")
    fo.close()


def RMSE(predictions, truth):
    return math.sqrt(sum((np.array(predictions) - np.array(truth))**2) / len(predictions))


def cross_validate(data, k):
    y = [float(d[2]) for d in data]
    rand_state = 42
    sum_error = 0
    for i in range(k):
        print(rand_state)
        data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=rand_state)
        recommender = ISMF(data_train, 50, 0.001, 0.1)
        recommender()
        test = [(d[0], d[1]) for d in data_test]
        predictions = recommender.predict(test)
        sum_error += RMSE(predictions, y_test)
        rand_state += 1
    return sum_error/k


class ISMF(object):

    def __init__(self, data, k, a, l):
        # initialise method parameters
        self._k = k
        self._alpha = a
        self._lambda = l

        # prepare data
        self.scores_by_user = defaultdict(dict)
        self.scores_by_artist = defaultdict(dict)
        self.train_set = set()

        for d in data:
            usr = d[0]
            art = d[1]
            score = float(d[2])
            self.scores_by_user[usr][art] = score
            self.scores_by_artist[art][usr] = score
            self.train_set.add((usr, art))

        # initialize P and Q matrices
        self.n = len(self.scores_by_user)
        self.m = len(self.scores_by_artist)
        rand_SEED = 66
        rand_max = 0.01
        np.random.seed(rand_SEED)
        p = np.random.rand(self.n, self._k)
        p = 2*rand_max*p - rand_max
        p[:, 0] = len(p[:, 0]) * [1]
        q = np.random.rand(self._k, self.m)
        q = 2*rand_max*q - rand_max
        q[1] = len(q[1]) * [1]

        self.p = {}
        for i, usr in enumerate(self.scores_by_user.keys()):
            self.p[usr] = p[i]
        self.q = {}
        for j, art in enumerate(self.scores_by_artist.keys()):
            self.q[art] = q[:, j]

    def __call__(self, validate=None):
        # gradient descent (until convergence or smth...)
        for iii in range(51):
            for u, a in self.train_set:
                # calculate e_ui
                e_ui = self.scores_by_user[u][a] - np.mean(list(self.scores_by_user[u].values())) - self.p[u].dot(self.q[a])
                #
                # # refresh p and q
                pu = self.p[u]
                self.p[u] = self.p[u] + self._alpha*(e_ui*self.q[a] - self._lambda*self.p[u])
                self.q[a] = self.q[a] + self._alpha*(e_ui*pu - self._lambda*self.q[a])
                self.p[u][0] = 1
                self.q[a][1] = 1

            # VALIDATE -> check rmse for current p, q
            if validate:
                test = [(d[0], d[1]) for d in validate]
                y_test = [float(d[2]) for d in validate]
                predictions = self.predict(test)
                rmse = RMSE(predictions, y_test)
                print("iter ", iii, ": ", rmse)


    def predict(self, test_data):
        ret = []
        for u, a in test_data:
            if u in self.scores_by_user.keys() and a in self.scores_by_artist.keys():
                r = self.p[u].dot(self.q[a]) + np.mean(list(self.scores_by_user[u].values()))
            elif u in self.scores_by_user.keys():
                # user is known, but artist is new (not in training data) --> use average score by user as prediction
                r = np.mean(list(self.scores_by_user[u].values()))
            elif a in self.scores_by_artist.keys():
                # artist is known, but user is new (not in training data) --> use average score given to artist
                r = np.mean(list(self.scores_by_artist[a].values()))
            else:
                # both user and artist are unknown --> use average score (1.82106569656) from the whole dataset
                r = 1.82106569656
            if r < 0:
                r = 0
            print(r)
            ret.append(r)
        return ret


if __name__ == "__main__":
    train = read_data("user_artists_training.dat")

    """data = [(t[0], t[1]) for t in train]
    y = [t[2] for t in train]
    cv_rmse = cross_validate(train, 4)
    print(cv_rmse)"""

    recommender = ISMF(train, 50, 0.001, 0.01)
    recommender()

    test = read_data("user_artists_test.dat")
    predictions = recommender.predict(test)
    write_to_file("out/naloga6.txt", predictions)
