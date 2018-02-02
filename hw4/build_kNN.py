'''
A script that implements a kNN algorithm. Features are words and their frequency, and it treats features as real valued.

For general kNN into & details:
https://saravananthirumuruganathan.wordpress.com/2010/05/17/a-detailed-introduction-to-k-nearest-neighbor-knn-algorithm/

Command to run: build_kNN.py training_data test_data k_val similarity_func sys_output_file > acc_file
'''

import sys
import argparse
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from helpful_functions import *


def make_kNN_vector(word_freq_data, vocab_list, vocab_set):
    '''
    Makes vectors out of data read in from an svm-light file based on a vocab (already read from that file)
    :param word_freq_data: a nested list of word_freq data in form [['a','11'],['the','20']...['cthulu','666']]. Pre-
    sorted alphabetically.
    :param vocab_list: a list of the vocabulary in the training data, from the first pass
    :param vocab_set: a set of the vocabulary in training data, same as vocab_list but for quick lookup
    :return: a vector
    '''
    word_freq_vector = np.zeros(len(vocab_list))
    # A loop to populate a word frequency vector with one position per vocab item with the frequency of the word
    # if it occurs in the training sample, and a 0 if it doesn't. In an extreme case where a training sample had every
    # word in the vocab, this would always increment i and j simultaneously

    # print('Training Length: {} Vocab length: {}'.format(len(word_freq_data), len(vocab_list)))
    i, j = 0, 0
    while j < len(word_freq_data):
        # if word is unknown, advance to next j
        if word_freq_data[j][0] not in vocab_set:
            j += 1
        # if words are equal, advance both indices
        elif vocab_list[i] == word_freq_data[j][0]:
            word_freq_vector[i] = int(word_freq_data[j][1])
            i += 1
            j += 1
        # if the vocab if earlier in the alphabet than the training data, advance only vocab index.
        # this is equivalent to elif vocab_list[i] < word_freq_data[j[0]], except that the word value at j should
        # always be >= the word value at i, so an else should be sufficient.
        else:
            i += 1
    return word_freq_vector


def svm_light_to_vec(train_filename, vocab_list, vocab_set):
    '''
    :param train_filename: svm-light training file, one training sample per line
    :param vocab_list: a list of the vocabulary in the training data, from the first pass
    :param vocab_set
    :return: a list of vectors as a numpy array, and a matching list of training gold labels, so the same index makes them match
    '''
    vector_list = [] # to store the vectors in, will be nested
    id2label = []  # to store training instance labels
    with open(train_filename, 'r') as infile:
        for line in infile:
            if line:
                sample = line.strip().split()
                label = sample[0]
                word_freq_data = sorted([x.split(':') for x in sample[1:]])  # [['a','11'],['b','12'],...]
                # id of the training record and index of the records vector in the nested list will match
                vector_list.append(make_kNN_vector(word_freq_data, vocab_list, vocab_set))
                id2label.append(label)
    return csr_matrix(vector_list), id2label


def get_vocab(train_filename):
    '''
    creates a sorted list of all vocab words from a given file
    :param train_filename: a training file in svm-light format, one record per line
    :return: a list, sorted alphabetically, of vocab words
    '''
    vocab_set = set()
    with open(train_filename, 'r') as infile:
        for line in infile:
            sample = line.strip().split()
            # add every word to a set
            [vocab_set.add(sample[idx].split(':')[0]) for idx in range(1, len(sample))]
    vocab_list = sorted(list(vocab_set))
    return vocab_list, vocab_set


def make_distribution(counter_obj):
    '''
    converts a counter into a distribution and returns a list
    :param counter_obj: in form 'thing': int
    :return: a list that is a distribution of it, in sorted order
    '''
    total = sum(counter_obj.values())
    dist = [(key, val/total) for key, val in counter_obj.items()]
    return sorted(dist, key=lambda x: x[1], reverse=True)


def run_knn(testing_vectors, training_vectors, training_labels, distance_measure, k_val):
    distributions, decisions = [], []
    # compare testing vector matrix with training vector matrix
    # these seem to return numpy arrays rather than sparse matrices
    if distance_measure == 1:
        results = euclidean_distances(testing_vectors, training_vectors)
        results = -results
    else:
        results = cosine_similarity(testing_vectors, training_vectors)

    # grab top k indices from each vector in results
    #[print('RESULTS {}'.format(res)) for res in results]
    for idx in range(testing_vectors.get_shape()[0]):
        row = results[idx]
        #print(idx, row)
        indices = np.argpartition(row, -k_val)[-k_val:]
        #print(indices)
        # grabs training labels by index, then puts into a counter to count their frequency
        votes = Counter([training_labels[i] for i in indices])
        # total votes also need to include votes with zero
        zero_votes = set(training_labels) - set(votes)
        for label in zero_votes:
            votes[label] = 0
        # convert the votes to a distribution and store them
        distributions.append(make_distribution(votes))
        # Most_common returns in sorted order as tuples
        winning_labels = votes.most_common()  # [('talk.politics.guns', 3), ('talk.politics.misc', 2)]
        # if first entry greater than second entry, then that is winner
        # print(winning_labels)
        if len(winning_labels) == 1:
            winner = winning_labels[0][0]
        # if there is more than one vote but no ties
        elif winning_labels[0][1] > winning_labels[1][1]:
            winner = winning_labels[0][0]
        else:
            # sort first by alphabetical order, then by value
            winning_labels.sort()
            winning_labels.sort(key=lambda x: x[1], reverse=True)
            #print('tie on {}'.format(idx))
            #print(winning_labels)
            #winning_labels_sorted = sorted(sorted(winning_labels), key=lambda x: x[1], reverse=True)
            winner = winning_labels[0][0]

        decisions.append(winner)

    return decisions, distributions


def write_sys_output(training_dist, training_labels, testing_dist, testing_labels, output_filename='tmp_output.txt'):
    '''
    :param training_dist: a nested list of tuples of format [[(label, prob), (label, prob)...], [(label, prob)...],...]
    :param training_labels: a list of gold labels, for printing
    :param testing_dist: same format as training
    :param testing_labels: same as training labels
    :param output_filename: the name of file to write results to
    :return: Nothing, writes to a file
    '''
    with open(output_filename, 'w') as outfile:
        for triple in [(training_dist, 'training', training_labels), (testing_dist, 'test', testing_labels)]:
            outfile.write('%%%%% {} data:\n'.format(triple[1]))
            for index in range(len(triple[0])):
                dist_str = ' '.join(['{} {:.5f}'.format(t[0], t[1]) for t in triple[0][index]])
                gold = triple[2][index]
                outfile.write('array:{} {} {}\n'.format(index, gold, dist_str))
            outfile.write('\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_filename', help='a file with training data, in svm-light format')
    parser.add_argument('test_filename', help='a file with testing data, in svm-light format')
    parser.add_argument('k_val', help='the number of nearest neighbours chosen for classification', type=int)
    parser.add_argument('similarity_func_id', help='Type of distance to use. 1: Euclidean, 2: Cosine', type=int)
    parser.add_argument('output_filename', help='name of file to write classification results to')
    args = parser.parse_args()
    #print(args.train_filename)
    # create vocab from training
    vocab_list, vocab_set = get_vocab(args.train_filename)
    # read in training file and record labeled instances as feature vectors
    train_vectors, training_labels = svm_light_to_vec(args.train_filename, vocab_list, vocab_set)
    classes_list = sorted(set(training_labels))
    #print([(vocab_list[i], i) for i in range(len(vocab_list))])
    #print(train_vectors)

    # for testing on d, find k training instances closest to d, and perform "voting"
    #first test on train
    print('Testing on training')
    training_results, training_dist = run_knn(train_vectors, train_vectors,
                                              training_labels, args.similarity_func_id, args.k_val)
    #then test on test
    print('Testing on testing')
    test_vectors, testing_labels = svm_light_to_vec(args.test_filename, vocab_list, vocab_set)
    testing_results, testing_dist = run_knn(test_vectors, train_vectors,
                                            training_labels, args.similarity_func_id, args.k_val)
    #write sys output
    write_sys_output(training_dist, training_labels, testing_dist, testing_labels, args.output_filename)

    # calc and print accuracies
    calc_accuracy(training_labels, training_results, classes_list, 'Training')
    calc_accuracy(testing_labels, testing_results, classes_list, 'Testing')

    """
    def old_run_knn(testing_vectors, training_vectors, training_labels, distance_measure, k_val):
        '''
        Runs knn on given test vectors, one at a time. calculate all distances from training vectors,
        sort and find k nearest neighbours, find majority vote label
        :param testing_vectors: an array of test vectors. Can be a single one but then still has to be in an array
        :param training_vectors: an array of training vectors
        :param training_labels: a dict mapping training vector indices to their labels
        :param distance_measure: 1 for Euclidean, else Cosine
        :param k_val: k nearest neighbours
        :return: list of decisions, in order, one per vectors, and a list of vote distributions (for determining confidence)
        also in order one per vector
        '''
        # for storing the labeling decisions, one per vector
        decisions, distributions = [], []

        for vector in testing_vectors:
            # get an array of distances or similarities
            if distance_measure == 1:
                results = euclidean_dist(vector, training_vectors)
                # make values negative to that the max returns the closest
                results = -results
            else:
                results = cosine_sim(vector, training_vectors)
            # grab top k indices
            indices = np.argpartition(results, k_val)[-k_val:]
            # grabs training labels by index, then puts into a counter to count their frequency
            votes = Counter([training_labels[i] for i in indices])
            # convert the votes to a distribution and store them
            distributions.append(make_distribution(votes))
            # Most_common returns in sorted order as tuples
            winning_labels = votes.most_common()  # [('talk.politics.guns', 3), ('talk.politics.misc', 2)]
            winner = None
            # if first entry greater than second entry, then that is winner
            #print(winning_labels)
            if len(winning_labels) == 1:
                winner = winning_labels[0][0]
            elif winning_labels[0][1] > winning_labels[1][1]:
                winner = winning_labels[0][0]
            # if not, grab all tied-winning entries, and set winner to first in alphabet
            else:
                winners = [winning_labels[0], winning_labels[1]]
                i = 2
                while i < len(winning_labels) and winning_labels[i][1] == winning_labels[0][1]:
                    winners.append(winning_labels[i][1])
                    i += 1
                    winner = sorted(winners)[0][0]

            decisions.append(winner)

        return decisions, distributions
    """