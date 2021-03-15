import csv
import math
from itertools import product
from itertools import combinations
import matplotlib.pyplot as plt


countries_receiving_votes = []
def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    # first pass trough CSV file --> discover countries that are participating (competing and/or voting)
    f = open(file_name, "rt", encoding="utf8")
    f.readline()
    countries_from = set()
    countries_to = set()
    min_year = 2222
    max_year = 0
    for l in csv.reader(f):
        if l[1] == "T":
            continue
        print(l)
        min_year = min(min_year, int(l[0]))
        max_year = max(max_year, int(l[0]))
        countries_from.add(l[2])
        countries_to.add(l[3])
    f.close()

    # standardization of countries names
    try:
        countries_from.remove("F.Y.R. Macedonia")
        countries_to.remove("F.Y.R. Macedonia")
    except:
        pass
    try:
        countries_from.remove("Serbia & Montenegro")
        countries_to.remove("Serbia & Montenegro")
    except:
        pass

    # output detected countries that are participating
    countries_from = list(sorted(countries_from))
    print(countries_from)
    print(len(countries_from))
    countries_to = list(sorted(countries_to))
    print(countries_to)
    print(len(countries_to))
    print(min_year, max_year)
    # save countries receiving votes for later use in hier. clustering class
    global countries_receiving_votes
    countries_receiving_votes = countries_to

    # initialize dictionary containing data by country (initialized to None for determining when a score is unavailable)
    data = {country: [None]*((max_year-min_year+1)*len(countries_to))
            for country in countries_from}

    # second pass trough CSV file --> read and save scores
    f = open(file_name, "rt", encoding="utf8")
    f.readline()
    for l in csv.reader(f):
        if l[1] == "T":
            continue
        year = int(l[0])
        from_country = l[2]
        if from_country == "F.Y.R. Macedonia":
            from_country = "North Macedonia"
        elif from_country == "Serbia & Montenegro":
            from_country = "Serbia"
        to_country = l[3]
        if to_country == "F.Y.R. Macedonia":
            to_country = "North Macedonia"
        elif to_country == "Serbia & Montenegro":
            to_country = "Serbia"
        score = int(l[4])
        data[from_country][(year - min_year)*len(countries_to) + countries_to.index(to_country)] = int(score)
    f.close()

    return data


def unpack(nested_list):
    """
    Unpack and flatten an arbitrarily nested list (of lists...).
    :return: generator that 'contains' all of the single items from the nested list
    """
    for item in nested_list:
        if isinstance(item, list):
            yield from unpack(item)
        else:
            yield item


def avg(nums):
    if len(nums) == 0:
        return 0
    else:
        return sum(nums)/len(nums)


"""def filtered_avg(nums):
    #f = [x for x in nums if x]
    #if len(f) == 0:
    #    return 0
    #return sum(f)/(len(f)**2)
    return avg([x for x in nums])"""


def out(elements, depth, branch):
    """
    Print a text dendrogram to the standard output.
    :param elements: nested list of lists representing the clustering - used with 'self.clusters'
    :param depth: current depth in the clustering structure, representing horizontal margin/offset
    :param branch: when true print 'branch' part of dendrogram, otherwise print one of the countries in the curr cluster
    """
    for e in elements:
        if type(e) is list and len(e) > 1:                                                                              # len(e) > 1 -> zato ker je v self.clusters itak vsak element/cluster svoj seznam, dolzine ena
            out(e, depth + 1, True)
            if branch:
                print("{}----|".format(4 * (depth - 1) * " "))
                branch = False
        else:
            print("{}---- {}".format(4 * depth * " ", e[0]))
            if branch:
                print("{}----|".format(4 * (depth - 1) * " "))
                branch = False


def find_dict_key(dic, s):
    """
    Takes key in form of a tuple or a single string and queries a dictionary for a matching key
    in the form of a nested tuple.
    :param dic: dictionary to be "queried"
    :param s: key that we are searching for
    :return: corresponding key in dictionary 'dic'
    """
    # prepare query to which we will compare keys from dict
    if type(s) == tuple:
        query = list(s)
    else:
        query = [s]

    # compare keys (and transform them to same form of a single list as the query) from dictionary
    for k in dic.keys():
        if type(k) == tuple:
            if type(k[0]) == tuple:
                curr = list(k[0])
            else:
                curr = [k[0]]
            if type(k[1]) == tuple:
                curr.extend(list(k[1]))
            else:
                curr.extend([k[1]])
        else:
            curr = [k]
        if query == curr:
            return k
    return None


#
#
# ----------------------------------------------------------------------------------------------------------------------
#       CLASS: HierarchicalClustering
# ----------------------------------------------------------------------------------------------------------------------
#
#
class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]
        # self.cluster_history is a dictionary containing all groupings that are made during the clustering process
        #   ==> keys = (cluster1, cluster2), values = (cumulative distance at which the clusters are grouped, location on dendrogram x_axis)
        self.cluster_history = {}

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        x_i = self.data[r1]
        x_j = self.data[r2]
        return math.sqrt(len(x_i)*avg([((a - b) ** 2)
                                       for a, b in zip(x_i, x_j)
                                       if a is not None and b is not None]))

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        cluster1 = unpack(c1)
        cluster2 = unpack(c2)
        return avg([self.row_distance(r1, r2)
                    for r1, r2 in product(cluster1, cluster2)                                                           # uporabim collections.product za kombinacijo vseh drzav iz obeh clustrov, ker metoda unpack vrne generator (zato ne deluje dvojni for loop)
                    if self.row_distance(r1, r2) > 0])

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        dist, pair = min((self.cluster_distance(c1, c2), (c1, c2))
                         for c1, c2 in combinations(self.clusters, 2)
                         if self.cluster_distance(c1, c2) != 0)
        return pair, dist

    def compare_cluster_to_others(self, cluster):
        # structure of the voting vector:  vector => (accumulated sum for a specific vote of the countries, number of votes)
        # example: [(sum_votes_for_country1_in_year X, num_of_votes_counted), (sum_votes_for_country2_in_year X, num_of_votes_counted), ...]
        # length of the vector equals the total length of the voting vectors in self.data
        # number of votes represents number of defined votes with which we will weight the sum of them
        print("comparing:")
        print(cluster)
        votes_cluster = [(0, 0) for _ in self.data["Germany"]]
        votes_other = [(0, 0) for _ in votes_cluster]
        # construct weighted average vector representing voting of all the countries in the cluster
        for country in cluster:
            votes_cluster = [(vote1[0] + vote2, vote1[1] + 1)
                             if vote2 is not None else (vote1[0], vote1[1])
                             for vote1, vote2 in zip(votes_cluster, self.data[country])]
        # weighted average vector representing all the other countries
        other_countries = list(set(self.data.keys()) - set(cluster))
        for country in other_countries:
            votes_other = [(vote1[0] + vote2, vote1[1] + 1)
                           if vote2 is not None else (vote1[0], vote1[1])
                           for vote1, vote2 in zip(votes_other, self.data[country])]

        # compare vectors and find outlier countries
        weighted_cluster = [sum_of_votes / num_of_votes
                            if num_of_votes > 0 else 0
                            for sum_of_votes, num_of_votes in votes_cluster]
        weighted_other = [sum_of_votes / num_of_votes
                          if num_of_votes > 0 else 0
                          for sum_of_votes, num_of_votes in votes_other]
        diff = [vote1 - vote2 for vote1, vote2 in zip(weighted_cluster, weighted_other)]
        cumulative_diff = {country: 0 for country in countries_receiving_votes}
        i = 0
        for d in diff:
            country_to = countries_receiving_votes[i]
            cumulative_diff[country_to] = cumulative_diff[country_to] + d
            if i < len(countries_receiving_votes) - 1:
                i = i + 1
            else:
                i = 0
        cumulative_diff_srtd = sorted(cumulative_diff, key=lambda k: abs(cumulative_diff[k]), reverse=True)
        best5 = dict(sorted(cumulative_diff.items(), key=lambda x: x[1], reverse=True)[:5])
        worst5 = dict(sorted(cumulative_diff.items(), key=lambda x: x[1])[:5])
        print(best5)
        print(worst5)

    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """


        while len(self.clusters) > 1:
            pair, dist = self.closest_clusters()
            print("--------------------")
            #print(self.clusters)
            print(str(dist) + " -> ")
            print(pair)
            self.clusters.remove(pair[0])
            self.clusters.remove(pair[1])
            self.clusters.append(list(pair))
            self.cluster_history[(tuple(country for country in unpack(pair[0])) if len(pair[0]) > 1 else pair[0][0],
                                  tuple(country for country in unpack(pair[1])) if len(pair[1]) > 1 else pair[1][0])] = (dist, 0)

        self.clusters = self.clusters[0]
        print("------------------------- CLUSTERING -----------------------------")
        print(self.clusters)
        print(self.cluster_history)
        print(self.data.keys())


    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        # plot the text dendrogram
        out([self.clusters], 0, False)

        countries = [""] + list(unpack(self.clusters)) + [""]
        idx = range(len(countries))
        # save initial countries and their indexes on the plot for use in visualisation algorithm
        for country in self.data:
            self.cluster_history[country] = (0, countries.index(country))

        # iterate over all clusterings and mark them on the plot
        for cluster in self.cluster_history:
            if type(cluster) == str:
                break
            c1 = cluster[0]
            c2 = cluster[1]
            # find correct dictionary keys for both clusters
            key1 = find_dict_key(self.cluster_history, c1)
            key2 = find_dict_key(self.cluster_history, c2)
            # read data (index/x_position & distance/y_position)
            x1 = self.cluster_history[key1][1]
            x2 = self.cluster_history[key2][1]
            y1 = self.cluster_history[key1][0]
            y2 = self.cluster_history[key2][0]
            y_big = max(y1, y2) + self.cluster_history[cluster][0]
            # update x_position and starting distance/y_position of the "combined" cluster (cluster <-- c1 + c2)
            self.cluster_history[cluster] = (y_big, (x1 + x2) / 2)
            # plot
            plt.plot([x1, x1], [y1, y_big], 'b-')
            plt.plot([x2, x2], [y2, y_big], 'b-')
            plt.plot([x1, x2], [y_big, y_big], 'b-')

        # setup and show plot
        plt.xticks(idx, countries, rotation='vertical')
        plt.tight_layout()
        plt.yticks(plt.yticks()[0], [round(tick/max(plt.yticks()[0]), 2) for tick in plt.yticks()[0]])
        plt.show()


if __name__ == "__main__":
    DATA_FILE = "eurovision-finals-1975-2019.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()
