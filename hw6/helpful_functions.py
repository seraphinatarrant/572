from sklearn.metrics import confusion_matrix
import numpy as np
import re
from math import exp
from collections import defaultdict

def calc_accuracy(gold_standard, predictions, classes, type):
    '''
    :param gold_standard: array of gold standard values
    :param predictions: array of predicted values
    :param classes: a list of class labels for labeling the columns and rows of the matric
    :param type: training or test
    :return: nothing - just prints
    '''
    header = 'Confusion matrix for the {} data:\nrow is the truth, column is the system output\n'.format(type.lower())
    confusion = confusion_matrix(gold_standard, predictions, labels=classes)
    correct, total = confusion.diagonal().sum(), confusion.sum()
    accuracy = correct/total
    max_char = max(map(len, classes))
    #print accuracy data
    print(header)
    print('{0:{1}} {2}'.format(' ', max_char, ' '.join(classes)))
    for index in range(len(classes)):
        print('{0:{1}} {2}'.format(classes[index], max_char+5, ' '.join([str(num) for num in confusion[index]])))
    print('\n{} accuracy={:.5f}\n\n'.format(type, accuracy))


def write_sys_output(dist, labels, type='test', output_filename='tmp_output.txt'):
    '''
    :param dist: a nested list of tuples of format [[(label, prob), (label, prob)...], [(label, prob)...],...]
    :param labels: a list of gold labels, for printing
    :param type: the type of data this is, either 'training' or test
    :param output_filename: the name of file to write results to
    :return: Nothing, writes to a file
    '''
    with open(output_filename, 'w') as outfile:
        #        for triple in [(training_dist, 'training', training_labels), (testing_dist, 'test', testing_labels)]:
        outfile.write('%%%%% {} data:\n'.format(type))
        for index in range(len(dist)):
            dist_str = ' '.join(['{} {:.5f}'.format(t[0], t[1]) for t in dist[index]])
            gold = labels[index]
            outfile.write('array:{} {} {}\n'.format(index, gold, dist_str))
        outfile.write('\n\n')

def read_maxent_model(model_file):
    '''
    MaxEnt model file is of this format:
    FEATURES FOR CLASS talk.politics.guns
    <default> -0.15835489728361463
    a -0.18898115373818014

    where FEATURES FOR CLASS declares the class, the default is the weight for the class, and below that are the feature
    weights for that class, until the next class declaration.
    :param model_file: the name of a file containing a maxent model
    :return: a feat2idx dict and a massive feature vector
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
    all_classes = list(class_weights)  # make classes a list so results can be a coindexed list, makes some processing easier
    all_features = set()  # create a set of all features to use in iteration
    [all_features.update(set(feature_weights[c])) for c in feature_weights]

    return class_weights, feature_weights, all_classes, all_features

def old_read_maxent_model(model_file):
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

def maxent_classify_standard(test_record, class_weights, feature_weights, all_classes, all_features, topN=None):
    '''
    Runs MaxEnt on a single test record based on the standard svm format. This version does not pick a winner,
    it only returns the confidences array
    :param test_record: alist [instanceName, gold label, set of features]
    :param class_weights: a dict of class: default class weight
    :param feature_weights: a nested dict of class: feature: weight
    :param topN: if present, returns only the topN probabilities/confidences (I really shouldn't keep switching terms)
    :return: sorted list of confidences (class probabilities) for all classes. Probabilities are in logform.
    '''
    inst_name, label, doc_features = test_record
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

    class_results = np.log10(((np.array(class_results)) / Z_total))
    #if topN declared, then keep only that many TODO CURRENTLY NOT WORKING
    if topN:
        indices = np.argpartition(class_results, -topN)[-topN:]
        class_results = [class_results[index] for index in indices]
    # get confidences
    class_probabilities = list(zip(all_classes, class_results))
    # Sort on the probabilities by alphabet and by confidence, with confidence most important
    class_probabilities.sort()
    class_probabilities.sort(key=lambda x: x[1], reverse=True)

    return class_probabilities


if __name__ == "__main__":
    print('Hello I am only a functions helper file! Love me anyway!')
    