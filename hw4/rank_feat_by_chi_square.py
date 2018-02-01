'''
A script that:
- reads in an input_file of svm-light feature vector form from stdin
- output_file has the format "featName score docFreq". The score is the chi-square score for
the feature; docFreq is the number of documents that the feature occurs in. The lines are sorted
by X^2 scores in descending order.
-X^2 contingency table treats features as binary - either present or not for a given sample of a class label

Command to run: cat input_file | rank_feat_by_chi_square.py > output_file
'''

import sys
import argparse
import fileinput
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import chisquare


def read_svm_light(infile):
    '''
    Will create a nested dict of {class_label: {word: num_samples}}, maintain a count of total samples per class
    :param infile: a file object with svm-light training data
    :return: the nested dict class_feature counts and the class_counts Counter
    '''
    class_counts, class_feature_counts = Counter(), defaultdict(Counter)
    all_features = set()
    for line in infile:
        sample = line.strip().split()  # ['label', 'word_1:freq_1, 'word_2:freq_2']
        class_label = sample[0]
        class_counts[class_label] += 1  #update class label counts
        # get all features in this sample, throw away frequency count as they are binary
        features = [pair.split(':')[0] for pair in sample[1:]]
        all_features.update(features)
        # update feature counts for this class label
        for feature in features:
            class_feature_counts[class_label][feature] += 1

    return class_feature_counts, class_counts, all_features


def rank_by_chi_squared(class_feature_counts, class_counts, all_features):
    '''

    :param class_feature_counts: nested dict by class by feature of the num class samples that have that feature
    :param class_counts: Counter of num samples in each class from training data
    :param all_features: set of all features from training data
    :return:
    '''
    feature_scores = []
    # force classes to be a list to ensure ordering for table comparison
    class_list, num_classes = list(class_counts), len(class_counts)
    total_samples = sum(class_counts.values())
    for feature in all_features:
        #initialise matrices
        o_matrix = np.zeros((2, num_classes))
        e_matrix = np.zeros((2, num_classes))
        # fill out observed data
        for i in range(num_classes):
            # count of samples for class with feature
            o_matrix[0][i] = class_feature_counts[class_list[i]][feature]
            # without feature is total for the class minus amount that has feature
            o_matrix[1][i] = class_counts[class_list[i]] - o_matrix[0][i]
        # fill out expected data: each cell for null hypothesis should be row_total*column_total / total_samples
        # could do this at the same time as o_matrix but looping twice shouldn't be very slow, and it's easier to calc
        # feature totals
        row_totals, col_totals = o_matrix.sum(axis=1), o_matrix.sum(axis=0)
        for j in range(len(row_totals)):
            for k in range(len(col_totals)):
                e_matrix[j][k] = row_totals[j] * col_totals[k] / total_samples
        score, p_val = chisquare(o_matrix, f_exp=e_matrix, axis=None)
        # o_matrix[0].sum() is the num docs the feature occurs in
        feature_scores.append([feature, score, int(o_matrix[0].sum())])

    return feature_scores


def print_results(results):
    '''
    Prints results sorted by chi squared values descending
    :param results: a nested list of [[feature, score, docfreq]...]
    :return: None
    '''
    [print('{} {:.5f} {}'.format(*feat_score)) for feat_score in sorted(results, key = lambda x:x[1], reverse=True)]


if __name__ == "__main__":
    # for testing - working version will read from stdin
    #infile = fileinput.input()
    infile = open(sys.argv[1], 'r')
    class_feature_counts, class_counts, all_features = read_svm_light(infile)
    results = rank_by_chi_squared(class_feature_counts, class_counts, all_features)
    print_results(results)



    