from unidecode import unidecode
import os
import math
from collections import Counter
import random
import matplotlib.pyplot as plt


def kmers(s, k=3):
    """Generates k-mers for an input string."""
    for i in range(len(s)-k+1):
        yield s[i:i+k]


class KMed:
    def __init__(self, path, mode="human-rights"):
        # dictionary of dictionaries, where each entry is a dictionary for some language, containing all the triplets
        #   (key) and their frequencies (value) --> dict(lang1: dict("abc": freq1, "bcd": freq2, ...), ...)
        self.data = {}
        # dictionary containing clusters, where the key is the name of the medoid and the value is a list of all the
        #   members of the cluster --> dict(medoid1: [list_of_cluster_members], medoid2: [list_of_cluster_members], ...)
        self.clusters = {}

        # selected languages and their file names
        #languages = ["slv", "src1", "src4", "src3", "rus", "ukr", "pql", "slo", "czc", "trk", "hng", "grk", "frn", "itn", "rum", "spn", "por", "eng", "ger", "dut", "swd", "fin", "nrn"]
        languages = ["slv", "src1", "src3", "rus", "ukr", "pql", "czc", "lit", "lat", "est", "chn", "hng", "grk", "frn",
                     "itn", "rum", "spn", "por", "eng", "ger", "dut", "swd", "nrn", "fin"]
        descript = ["slovenščina", "bosanščina", "bosanščina (cirilica)", "srbščina", "ruščina", "slovaščina", "češčina", "albanščina", "turščina", "kitajščina", "grščina",
                       "francoščina", "italjanščina", "romunščina", "španščina", "portugalščina", "angleščina", "nemščina", "danščina", "nizozemščina", "švedščina", "finščina", "norveščina"]
        self.description = {"slv": "slovenščina", "src1": "bosanščina", "src4": "bosanščina (cirilica)", "src3": "srbščina", "rus": "ruščina",
                            "ukr": "ukrajinščina", "pql": "poljščina", "slo": "slovaščina", "czc": "češčina", "aln": "albanščina",
                            "lit": "litvanščina", "lat": "latvijščina", "est": "estonščina", "trk": "turščina", "hng": "madžarščina", "grk": "grščina",
                            "frn": "francoščina", "itn": "italjanščina", "rum": "romunščina", "spn": "španščina", "por": "portugalščina",
                            "eng": "angleščina", "ger": "nemščina", "dut": "nizozemščina", "swd": "švedščina", "fin": "finščina", "nrn": "norveščina"}

        # if we are using news articles (mode == "news"), change the path to be the 'news' folder
        if mode == "news":
            path = "news/"

        # read data
        for file in os.listdir(path):
            if mode == "human-rights" and file[:-4] not in languages:
                continue
            print(file)
            name = os.path.splitext(os.path.basename(file))[0]
            text = " ".join([line.strip() for line in open(path+file, "rt", encoding="utf8").readlines()])
            text = unidecode(text).lower()
            text = " ".join(text.split())                                                                               # remove multiple whitespaces which occur from newlines in raw text
            print(text)
            self.data[name] = dict(Counter(kmers(text, 3)))

        print(self.data)
        print("-------- end init ---------------------------------------------------------------------------------")

    """def cosss(self, x, y):
        return sum(x_i * y_i for x_i, y_i in zip(x, y)) / (math.sqrt(sum(x_i**2 for x_i in x)) * math.sqrt(sum(y_i**2 for y_i in y)))

    def cos_test(self, x, y):
        shared_keys = set(x.keys()) & set(y.keys())
        up = sum(x[key] * y[key] for key in shared_keys)
        down = math.sqrt(sum(x_i ** 2 for x_i in x.values())) * math.sqrt(sum(y_i ** 2 for y_i in y.values()))
        return up / down"""

    def cos(self, lang1, lang2):
        """
        Calculate cosine distance (distance = 1 - similarity) between two languages.
        :param lang1: name of first language
        :param lang2: name of second language
        :return: cosine distance
        """
        x = self.data[lang1]
        y = self.data[lang2]
        shared_keys = set(x.keys()) & set(y.keys())
        up = sum(x[key] * y[key] for key in shared_keys)
        down = math.sqrt(sum(x_i**2 for x_i in x.values())) * math.sqrt(sum(y_i**2 for y_i in y.values()))
        return 1 - up/down

    def cos_unknown(self, lang1, unknown):
        """
        Calculate cosine distance between a language in self.dict and text in an unknown language.
        """
        x = self.data[lang1]
        y = unknown
        shared_keys = set(x.keys()) & set(y.keys())
        up = sum(x[key] * y[key] for key in shared_keys)
        down = math.sqrt(sum(x_i**2 for x_i in x.values())) * math.sqrt(sum(y_i**2 for y_i in y.values()))
        return 1 - up/down

    def find_closest_cluster(self, lang):
        """
        Find the cluster that is closest to the language 'lang'.
        :param lang: language for which we are searching the closest cluster
        :return: name of the closest cluster's medoid
        """
        dist, medoid = min((self.cos(lang, m), m)
                           for m in self.clusters.keys())
        return medoid

    def recognize_language(self, file_path):
        """
        Determine which language the text in 'file_path' belongs to.
        :param file_path: path to the file that contains the text
        :return: three most likely languages and the probabilities for each of them
        """
        # preparing the data
        text = " ".join([line.strip() for line in open(file_path, "rt", encoding="utf8").readlines()])
        text = unidecode(text).lower()
        text = " ".join(text.split())  # remove multiple whitespaces which occur from newlines in raw text

        # construct vector of frequencies for triplets of letters in text 'file_path'
        unknown_language = dict(Counter(kmers(text, 3)))

        # find 3 closest languages and their distances
        distances = {}
        for language in self.data:
            dist = self.cos_unknown(language, unknown_language)
            distances[language] = dist

        # transform distances (d = float number [0, inf]) into probabilities (p = float [0.0, 1.0])
        inv_distances = {lang: 1 / distances[lang]**2 for lang in distances}
        normalisation_factor = sum(inv_distances.values())
        probabilities = {lang: inv_distances[lang] / normalisation_factor for lang in inv_distances}
        """sum_of_all_prob = sum(val for val in probabilities.values())
        print("probability sum", sum_of_all_prob)
        print("All the probabilities", probabilities)"""
        top3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        print("The 3 most likely languages (that the text {} is written in) and their probabilities are: {} - {:.2%}, {} - {:.2%}, {} - {:.2%}.".format(file_path, self.description[top3[0][0]], top3[0][1], self.description[top3[1][0]], top3[1][1], self.description[top3[2][0]], top3[2][1]))
        return top3

    def silh(self, lang):
        """
        Calculate silhouette for an individual example/case (language).
        """
        # prepare cluster that contains 'lang'
        cluster = []
        for med in self.clusters:
            if lang == med:
                cluster = self.clusters[med][:]
                break
            if lang in self.clusters[med]:
                cluster = self.clusters[med][:]
                cluster.remove(lang)
                cluster.append(med)
                break
        # average distance to languages in same cluster
        #a = sum(self.cos(lang, c) for c in cluster) / len(cluster) if len(cluster) else 0
        if len(cluster) > 0:
            a = sum(self.cos(lang, c) for c in cluster) / len(cluster)
        else:
            return 0

        # find other cluster (that doesn't contain 'lang') that is closest to 'lang'
        other = []
        min_b = 99999
        for med in self.clusters:
            if lang == med or lang in self.clusters[med]:
                continue
            else:
                other = self.clusters[med][:]
                other.append(med)
                b_j = sum(self.cos(lang, o) for o in other) / len(other)
                if b_j < min_b:
                    min_b = b_j
        # average distance to languages in closest other cluster (doens't contain 'lang')
        b = min_b

        return (b - a) / max(a, b)

    def silhouette(self):
        """
        Calculate silhouette of the entire clustering.
        """
        s = [self.silh(lang) for lang in self.data]
        return sum(s) / len(s)

    def out(self):
        print(self.data)

    def rand_init(self, k=5):
        medoids = random.sample(self.data.keys(), k)
        self.clusters = {m: [] for m in medoids}
        #print(medoids)
        #print(self.clusters)

    def k_med_clustering(self):
        """
        K-medoid clustering algorithm (default k=5, if needed provide different 'k' to self.rand_init).
        :return: dictionary containing resulting clusters (dict_key = cluster medoid, dict_val = other members of cluster)
        """
        # initialize the starting medoids by randomly selecting k=5 languages
        self.rand_init()

        # main loop - find a "better" medoid and repeat until a more representative medoid can't be found anymore
        while True:
            # assign all languages to apropriate cluster (shortest distance to the medoid)
            for language in self.data:
                if language in self.clusters.keys():
                    continue
                closest = self.find_closest_cluster(language)
                self.clusters[closest].append(language)

            min_cost = 99999
            min_lang = None
            # determine new medoid
            for med in self.clusters:
                # current cost
                if len(self.clusters[med]) > 0:
                    curr_cost = sum(self.cos(med, lang) for lang in self.clusters[med])
                else:
                    continue
                for other in self.clusters[med]:
                    # cost when medoid is switched with language 'other' (from the same cluster)
                    switched = self.clusters[med][:]
                    switched.remove(other)
                    switched.append(med)
                    switched_cost = sum(self.cos(other, lang) for lang in switched)
                    # if the cost of the switched medoid is lower than the cost of the curr. medoid -> remember new min
                    if switched_cost < curr_cost and switched_cost < min_cost:
                        min_cost = switched_cost
                        min_lang = other
                        old_medoid = med

            # if there is a new medoid found, switch the old medoid with the new, otherwise stop and return the clustering
            if min_lang is not None:
                self.clusters.pop(old_medoid)
                self.clusters[min_lang] = []
                for med in self.clusters:
                    self.clusters[med] = []
            else:
                return self.clusters.copy()

    def run(self):
        all_silhouettes = []
        min_silh = 99999
        worst = None
        max_silh = 0
        best = None
        for i in range(100):
            clustering_i = self.k_med_clustering()
            silh_i = self.silhouette()
            all_silhouettes.append(silh_i)
            if silh_i > max_silh:
                max_silh = silh_i
                best = clustering_i
            if silh_i < min_silh:
                min_silh = silh_i
                worst = clustering_i

        # print the clustering with best/worst silhouette
        print("najbolši clustering =", best)
        print("silhueta razbitja =", max_silh)
        print("najslabši clustering =", worst)
        print("silhueta razbitja =", min_silh)

        # plot histogram showing silhouette values over all the iterations
        plt.hist(all_silhouettes, 20)
        plt.xlabel("vrednost silhuete")
        plt.ylabel("frekvenca")
        plt.show()

        print("---------- recognize language --------------")
        lang_file = "test_recognize/slovenian.txt"
        self.recognize_language(lang_file)
        lang_file = "test_recognize/russian.txt"
        self.recognize_language(lang_file)
        lang_file = "test_recognize/spanish.txt"
        self.recognize_language(lang_file)
        lang_file = "test_recognize/french.txt"
        self.recognize_language(lang_file)
        lang_file = "test_recognize/german.txt"
        self.recognize_language(lang_file)
        lang_file = "test_recognize/english.txt"
        self.recognize_language(lang_file)
        lang_file = "test_recognize/greek.txt"
        self.recognize_language(lang_file)


if __name__ == "__main__":
    path_to_file = "ready/"
    km = KMed(path_to_file)
    km.run()
