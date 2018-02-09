'''
A script that classifies test data given a MaxEnt model learned from training data. This script requires a model file
to run - i.e. it does not generate one from training data.
It outputs classification results and accuracy.

Command to run: maxent_classify.py test_data model_file sys_output > acc_file
'''

import argparse
from collections import defaultdict
from math import exp
import numpy as np
import re
from helpful_functions import calc_accuracy, write_sys_output


def svm_light_to_binary_features(input_file):
    '''
    Reads in training or test data and converts to a list of pairs of label and the feature set from that sample
    :param input_file: name of a file in svm-light format
    :return: a nested list of form [[class_label, set of binary features present in sample],...]
    '''
    label_feature_set = []
    with open(input_file, 'r') as infile:
        for line in infile:
            if line:
                sample = line.strip().split()
                label = sample[0]
                doc_set = set()
                for index in range(1, len(sample)):
                    word, count = sample[index].split(':')
                    doc_set.add(word)
                label_feature_set.append([label, doc_set])

    return label_feature_set


def read_maxent_model(model_file):
    '''
    MaxEnt model file is of this format:
    FEATURES FOR CLASS talk.politics.guns
    <default> -0.15835489728361463
    a -0.18898115373818014

    where FEATURES FOR CLASS declares the class, the default is the weight for the class, and below that are the feature
    weights for that class, until the next class declaration.
    :param model_file: the name of a file containing a maxent model
    :return: a class_weights dict, of form {class: default weight} and a feature_weights dict, nested, of form
    {class: {feature: weight} }
    '''
    class_declaration = 'FEATURES FOR CLASS '
    default_declaration = '<default> '
    class_regex = re.compile('(?<=^{} ).+'.format(class_declaration))
    default_regex = re.compile('(?<=^{} ).+'.format(default_declaration))
    class_weights = defaultdict(float)
    feature_weights = defaultdict(lambda: defaultdict(float))
    with open(model_file, 'r') as mfile:
        for line in mfile:
            if line:
                line = line.strip()
                if line.startswith(class_declaration):
                    #current_class = class_regex.search(line).group(0)  # extracts the class label following declaration
                    current_class = re.search(r'(?<=^FEATURES FOR CLASS ).+', line).group(0)
                    continue
                if line.startswith(default_declaration): # not the fastest, but readable. could collapse into a single if statement if an issue.
                    #class_weights[current_class] = float(default_regex.search(line).group(0))
                    class_weights[current_class] = float(re.search(r'(?<=^<default> ).+', line).group(0))
                    continue

                feature, weight = line.split()
                feature_weights[current_class][feature] = float(weight)

    return class_weights, feature_weights


def maxent_classify(test_records, class_weights, feature_weights):
    '''
    Runs MaxEnt an array of test records
    :param test_records: a nested list [[gold label, set of features],...]
    :param class_weights: a dict of class: default class weight
    :param feature_weights: a nested dict of class: feature: weight
    :return: 3 lists - one of gold labels, one of predictions, and the final of the confidences
    '''
    gold_labels, predictions, confidences = [], [], []
    all_classes = list(class_weights)  # make classes a list so results can be a coindexed list, makes some processing easier
    all_features = set()  # create a set of all features to use in iteration
    [all_features.update(set(feature_weights[c])) for c in feature_weights]

    for record in test_records:
        label, doc_features = record
        features = doc_features & all_features
        class_results = []  # will have the results for the record for each class, where indices for class_list and results are the same
        Z_total = 0

        for class_label in all_classes:
            sum = class_weights[class_label]
            for feature in features:
                sum += feature_weights[class_label][feature]
            sum = exp(sum)
            Z_total += sum
            class_results.append(sum)

        class_results = (np.array(class_results)) / Z_total
        # get confidences
        class_probabilities = list(zip(all_classes, class_results))
        # Sort on the probabilities by alphabet and by confidence, with confidence most important
        class_probabilities.sort()
        class_probabilities.sort(key=lambda x: x[1], reverse=True)
        confidences.append(class_probabilities)
        # get prediction given confidences. np argmax returns the max index, which can be used to retrieve the class
        predictions.append(all_classes[np.argmax(class_results)])
        #store gold label for accuracy calculations later
        gold_labels.append(label)



    return gold_labels, predictions, confidences




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('test_filename', help='A test file in svm-light format')
    p.add_argument('model_file', help='A MaxEnt .txt model file, of form "feature weight" one feature per line')
    p.add_argument('output_filename', help='the name for the sys out file')

    args = p.parse_args()
    # Read in model
    c_weights, f_weights = read_maxent_model(args.model_file)
    # Read in test document
    test_records = svm_light_to_binary_features(args.test_filename)
    # classify test records
    gold, predict, conf = maxent_classify(test_records, c_weights, f_weights)
    #TODO sort out what's going on with calc_accuracy
    calc_accuracy(gold, predict, sorted(list(c_weights)), 'Testing')
    write_sys_output(conf, gold, 'test', args.output_filename)

    