import csv
import gzip
import lpputils
import linear
import numpy as np


# FUN: Read raw data from input file.
def read_raw_data(file_name):
    f = gzip.open(file_name, "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    data = [d for d in reader]
    return data


# FUN: Calculate Mean Absolute Error for scoring.
def MAE(predictions, true):
    return sum([np.abs(p - r) for p, r in zip(predictions, true)]) / len(predictions)


def weekday_marker(wday):
    markers = np.zeros(7)
    markers[wday] = 1
    return list(markers)


def day_marker(day):
    markers = np.zeros(31)
    markers[day-1] = 1
    return list(markers)


def hour_marker(hour):
    markers = np.zeros(24)
    markers[hour] = 1
    return list(markers)


def minute_marker(minute):
    markers = np.zeros(60)
    markers[minute] = 1
    return list(markers)


def is_date_between(date, date1, date2):
    d1 = lpputils.parsedate(date1)
    d2 = lpputils.parsedate(date2)
    if (date.month >= d1.month and date.day >= d1.day) and (date.month <= d2.month and date.day <= d2.day):
        return [1]
    else:
        return [0]


def is_holiday(date):
    major_holidays = [lpputils.parsedate("2012-01-01 00:00:00.000"), lpputils.parsedate("2012-01-02 00:00:00.000"), lpputils.parsedate("2012-02-08 00:00:00.000"),
                      lpputils.parsedate("2012-04-08 00:00:00.000"), lpputils.parsedate("2012-04-09 00:00:00.000"),
                      lpputils.parsedate("2012-05-01 00:00:00.000"), lpputils.parsedate("2012-05-02 00:00:00.000"), lpputils.parsedate("2012-06-25 00:00:00.000"),
                      lpputils.parsedate("2012-08-15 00:00:00.000"), lpputils.parsedate("2012-11-01 00:00:00.000"), lpputils.parsedate("2012-12-26 00:00:00.000")]
    d = lpputils.parsedate(date)
    for holiday in major_holidays:
        if d.month == holiday.month and d.day == holiday.day:
            return [1]
    return [0]


def datetime_to_float(date):
    month_lengths = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = sum(month_lengths[:date.month]) + date.day
    return (days*24 + date.hour) / (366*24)


def time_of_day(date):
    minutes = date.hour + date.minute
    return minutes / (24 * 60)


def season_marker(date):
    if date.month in [3,4,5]:
        return [1, 0, 0, 0]
    elif date.month in [6,7,8]:
        return [0, 1, 0, 0]
    elif date.month in [9,10,11]:
        return [0, 0, 1, 0]
    elif date.month in [1,2,12]:
        return [0, 0, 0, 1]


# FUN: Process the raw input - form new 0/1 variables, etc.
def filter_input(inp):
    return [datetime_to_float(lpputils.parsedate(inp[6]))] + season_marker(lpputils.parsedate(inp[6])) + weekday_marker(lpputils.parsedate(inp[6]).weekday()) + hour_marker(lpputils.parsedate(inp[6]).hour) + minute_marker(lpputils.parsedate(inp[6]).minute) + is_holiday(inp[6]) + is_date_between(lpputils.parsedate(inp[6]), "2012-06-23 00:00:00.000", "2012-09-01 00:00:00.000")


# ----------------------------------------------------------------------------------------------------------------------
# | TRAIN & TEST/VALIDATE: Create model/classifier and test performance by splitting the training data.
# ----------------------------------------------------------------------------------------------------------------------
def train_and_test():
    raw = read_raw_data("train_pred.csv.gz")

    filtered_data = np.array([filter_input(d) for d in raw])
    travel_time = np.array([lpputils.tsdiff(d[8], d[6]) for d in raw])

    lr = linear.LinearLearner(lambda_=1.)
    napovednik = lr(filtered_data[:8000], travel_time[:8000])

    predictions = [napovednik(e) for e in filtered_data[8000:]]
    print(MAE(predictions, travel_time[8000:]))


# ----------------------------------------------------------------------------------------------------------------------
# | TRAIN THE MODEL: Create model/classifier from training data.
# ----------------------------------------------------------------------------------------------------------------------
def train():
    raw = read_raw_data("train_pred.csv.gz")

    filtered_data = np.array([filter_input(d) for d in raw])
    travel_time = np.array([lpputils.tsdiff(d[8], d[6]) for d in raw])

    lr = linear.LinearLearner(lambda_=1.)
    napovednik = lr(filtered_data, travel_time)

    return napovednik


# ----------------------------------------------------------------------------------------------------------------------
# | PREDICT ON TEST DATA: Create model and write predictions for test data to a file.
# ----------------------------------------------------------------------------------------------------------------------
def write_to_file(file_name, l_classifier):
    f = gzip.open("test_pred.csv.gz", "rt")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    test_data = [d for d in reader]
    filtered_test = np.array([filter_input(d) for d in test_data])

    predictions = [l_classifier(e) for e in filtered_test]

    fo = open(file_name, "wt")
    for x, y in zip(test_data, predictions):
        print(lpputils.tsadd(x[6], y))
        fo.write(lpputils.tsadd(x[6], y) + "\n")
    fo.close()


if __name__ == "__main__":

    print("PREDTEKMOVANJE - starting training")
    write_to_file("predtekmovanje-results.txt", train())
    print("predictions & file created..... END")

    #train_and_test()

