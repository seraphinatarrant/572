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
    :param type: training or test or baseline
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


def svm_light_to_binary_TDL_features(input_file, dummy_label=None):
    '''
    Reads in training or test data and converts to a list of gold label, dummy label annotation, and the feature set
    from that sample.
    Also returns a list of all classes and of all features.
    :param input_file: name of a file in svm-light format
    :param dummy_label:
    :return: a nested list of form [[gold class_label, dummy class label, set of binary features present in sample],...]
    and a list of all class_labels and of all features. The first class in the list of class labels is guaranteed to be
    the first label encountered.
    '''
    label_feature_list = []
    if dummy_label:  # if dummy label is explicitly provided, use it instead of the first label
        first, first_label = False, dummy_label
    else:
        first, first_label = True, None
    all_labels, all_features = set(), set()
    with open(input_file, 'r') as infile:
        for line in infile:
            if line:
                sample = line.strip().split()
                label = sample[0]
                if first:
                    first_label = label
                    first = False
                doc_set = set()
                for index in range(1, len(sample)):
                    word, count = sample[index].split(':')
                    doc_set.add(word)
                label_feature_list.append([label, first_label, doc_set])
                all_labels.update([label])
                all_features.update(doc_set)
    # wizardry so I easily get the first label at the first index. If there were a gazillion class labels it would be
    # slow, but there are max 50 in the cases we're doing so it's ok.
    all_labels = list(all_labels)
    all_labels.remove(first_label)
    all_labels.insert(0, first_label)
    # note that since I do no sort these lists, and first store them as sets then as lists, the ordering of the lists
    # is nondeterministic. So if I later use these in a way that depends on the ordering, the results may also be non-
    #deterministic
    return label_feature_list, all_labels, list(all_features)


if __name__ == "__main__":
    print('Hello I am only a functions helper file! Love me anyway!')
    