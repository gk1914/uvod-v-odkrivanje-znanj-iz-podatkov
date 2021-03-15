import csv
import gzip
from collections import defaultdict

import lpputils
import linear
import numpy as np


def read_raw_data(file_name):
    f = gzip.open(file_name, "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    data = [d for d in reader]
    return data


def read_weather():
    f = open("weather-arso-LJ-2012.csv", "rt")
    reader = csv.reader(f, delimiter=",")
    next(reader)
    #data = [d for d in reader]
    data = {(int(d[2][5:7]), int(d[2][8:10])): [int(float(d[4]) > 2), int(float(d[4]) > 10), int(float(d[5]) >= 2)] for d in reader if len(d)}
    print(data)
    return data


def weather_marker(w, date):
    m = lpputils.parsedate(date).month
    d = lpputils.parsedate(date).day
    return w[(m, d)]


def month_marker(month, day):
    markers = np.zeros(12)
    markers[month - 1] = 1
    return [month/12]


def datetime_to_float(date):
    month_lengths = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = sum(month_lengths[:date.month]) + date.day
    return (days*24 + date.hour) / (366*24)


def filter_input(inp, weather):
    wday = lpputils.parsedate(inp[6]).weekday()/7
    hour = lpputils.parsedate(inp[6]).hour/24
    minute = lpputils.parsedate(inp[6]).minute/60
    dtime = datetime_to_float(lpputils.parsedate(inp[6]))
    return [wday, wday**2, wday**3, wday**4, wday**5, wday**6, wday**7, hour, hour**2, hour**3, hour**4, hour**5, hour**6, hour**7, hour**8, hour**9, hour**10, hour**11, hour**12, hour**13, minute, minute**2, minute**3, minute**4, minute**5, dtime, dtime**2, dtime**3] + weather_marker(weather, inp[6])


def MAE(predictions, true):
    return sum([np.abs(p - r) for p, r in zip(predictions, true)]) / len(predictions)


# ----------------------------------------------------------------------------------------------------------------------
# | SEPARATE BY MONTH: Separate data by month for scoring/cross validation.
# ----------------------------------------------------------------------------------------------------------------------

def is_month(month, d):
    return lpputils.parsedate(d).month == month


def read_by_month(file_name, month):
    f = gzip.open(file_name, "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    data_by_month = defaultdict(list)
    for d in reader:
        data_by_month[lpputils.parsedate(d[6]).month].append(d)
    return data_by_month


def CV_month(month, month_seperated, weather):
    raw = []
    for i in range(12):
        if i+1 != month:
            raw.extend(month_seperated[i+1])
    test_set = [filter_input(t, weather) for t in month_seperated[month]]
    true_travel = [lpputils.tsdiff(t[8], t[6]) for t in month_seperated[month]]
    print("xd")

    l = SeparateBySetLearner(linear.LinearLearner(lambda_=1))
    c = l(raw, weather)
    predictions = [c(e, weather) for e in month_seperated[month]]
    print(MAE(predictions, true_travel))

def teeest():
    weather = read_weather()
    raw = read_by_month("train.csv.gz", 2)
    CV_month(4, raw, weather)


# ----------------------------------------------------------------------------------------------------------------------
# | SEPARATE BY (BUS) LINE: Create separate models for individual bus routes (based on bus ID, route direction).
# ----------------------------------------------------------------------------------------------------------------------

def linekey(d):
    return tuple(d[2:5])


class SeparateBySetLearner(object):

    def __init__(self, base):
        self.base = base

    def __call__(self, data, weather):
        rsd = defaultdict(list)
        ys = defaultdict(list)
        rsc = {}
        #separate different bus lines
        for d in data:
            rsd[linekey(d)].append(filter_input(d, weather))
            ys[linekey(d)].append(lpputils.tsdiff(d[8], d[6]))

        #build a prediction model for each line
        #with self.base
        for k in rsd:
            cl = self.base(np.array(rsd[k]), np.array(ys[k]))
            rsc[k] = cl
        return SeparateBySetClassifier(rsc)


class SeparateBySetClassifier(object):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def __call__(self, x, weather):
        #pass to input to the correct classifier for that line
        try:
            return self.classifiers[linekey(x)](filter_input(x, weather))
        except:
            # a new line: make an average
            return np.mean([c(filter_input(x, weather)) for c in self.classifiers.values()])


# ----------------------------------------------------------------------------------------------------------------------
# | PREDICT ON TEST DATA: Create model and write predictions for test data to a file.
# ----------------------------------------------------------------------------------------------------------------------

def write_to_file(file_name, l_classifier, weather):
    f = gzip.open("test.csv.gz", "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    test_data = [d for d in reader]

    predictions = [l_classifier(d, weather) for d in test_data]

    fo = open(file_name, "wt")
    for x, y in zip(test_data, predictions):
        print(lpputils.tsadd(x[6], y))
        fo.write(lpputils.tsadd(x[6], y) + "\n")
    fo.close()


if __name__ == "__main__":
    #teeest()

    raw = read_raw_data("train.csv.gz")
    weather = read_weather()
    l = SeparateBySetLearner(linear.LinearLearner(lambda_=5))
    c = l(raw, weather)
    write_to_file("poly.txt", c, weather)
