'''
A script that builds a Multinomial Naive Bayes model classifier, given some training data, and then tests it
on both training and test data. A Multinomial Naive Bayes model has real valued input for word frequencies, but unlike
Bernoulli, does not encode absence of the word feature.
Command to run: build_NB2.sh training_data test_data class_prior_delta cond_prob_delta model_file sys_output > acc file

Calculation of class priors is the same as Bernoulli, but calculation of conditional word probabilties and
classification differs.
'''

import sys
from collections import Counter, defaultdict
from math import log10, pow
import numpy as np
from sklearn.metrics import confusion_matrix
from helpful_functions import calc_accuracy


def train_NB_Multinomial(training_filename, class_prior_delta=1.0, cond_prob_delta=1.0):
    '''
    :param training_filename: name of file with training data in svm-light format
    :param class_prior_delta: A hyperparameter, the delta value for smoothing priors
    :param cond_prob_delta: A hyperparameter, the delta value for smoothing conditional probabilities
    :return: dict of prior probabilities of form class_label: (prob, logprob) and a nested dict of conditional
    probabilties of form class_label: word: (prob, logprob). Also returns the processed training data in form of a
    list of nested lists (one per sample) of form [true label, list of tuples of form (word, freq)] for each word in a given
    training sample
    '''
    #at training stage, build a model with class prior probabilities, and with conditional probabilities (word|class).
    # defaults to add-one smoothing if no deltas are provided
    class_counts, word_class_counts = Counter(), Counter()
    priors, conditional_probs = defaultdict(tuple), defaultdict(lambda: defaultdict(tuple))
    training_records_list = []
    Z_totals = Counter() #for efficiency gains with cond pob calculations
    words = set()
    with open(training_filename, 'rU') as infile:
        for line in infile:
            sample = line.strip().split()
            class_label = sample[0]
            class_counts[class_label] += 1
            #loop through training data and populate counters. Also store training data for later testing.
            doc_dict = Counter()
            for index in range(1, len(sample)):
                word, count = sample[index].split(':')
                count = int(count)
                word_class_counts[(word, class_label)] += count
                Z_totals[class_label] += count
                words.add(word)
                doc_dict[word] = count
            training_records_list.append([class_label, doc_dict]) #list of label, set of words in that doc
    #calculate prior probabilities
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts.keys())
    for key in class_counts:
        class_prior = (class_counts[key] + class_prior_delta) / (total_samples + num_classes*class_prior_delta)
        priors[key] = (class_prior, log10(class_prior))
    #calculate conditional probabilities
    for key in word_class_counts:
        word, class_label = key
        # count of (word,class) over count (class). divide by |V|*cond_prob_delta to smooth
        cond_prob = (word_class_counts[key] + cond_prob_delta) / (Z_totals[class_label] + cond_prob_delta*len(words))
        #print(key, word_class_counts[key])
        conditional_probs[class_label][word] = (cond_prob, log10(cond_prob))
    #calculate smoothing values for each class and each word given class
    class_smooth_val = class_prior_delta / (total_samples + num_classes*class_prior_delta)
    word_class_smooth = defaultdict(tuple)
    for key in class_counts:
        #word class smooth is dependent only on class
        smooth_prob = cond_prob_delta / (Z_totals[class_label] + cond_prob_delta*len(words))
        word_class_smooth[key] = (smooth_prob, log10(smooth_prob))

    return priors, conditional_probs, class_smooth_val, word_class_smooth, training_records_list, words


def output_model(output_filename, class_priors, conditional_probs):
    with open(output_filename, 'w') as outfile:
        #write priors
        outfile.write('%%%%% prior prob P(c) %%%%%\n')
        for key, value in sorted(class_priors.items()):
            outfile.write('{} {:.5f} {:.5f}\n'.format(key, value[0], value[1]))
        outfile.write('%%%%% conditional prob P(f|c) %%%%%\n')
        for class_label in sorted(conditional_probs):
            outfile.write('%%%%% conditional prob P(f|c) c={} %%%%%\n'.format(class_label))
            for word, probs in sorted(conditional_probs[class_label].items()):
                outfile.write('{} {} {:.5f} {:.5f}\n'.format(word, class_label, probs[0], probs[1]))

def test_NB_Multinomial(sample, class_priors, cond_probs, class_label_list,
                        all_vocab, class_smooth, word_given_class_smooth):
    '''
    :param sample: list of 0th element as gold label, 1st element a Counter of word:freq where freq is num times in doc
    :param class_priors: dict of class prior probs of form label: (prob, logprob)
    :param cond_probs: a nested dict of conditional probabilities of form class: word: (prob, logprob)
    :param class_label_list: a list of all class labels. Same a class_priors.keys(), but with fixed order
    :param all_vocab: a set of all words in the vocab. For filtering words that are in the sample but are unknown
    :param class_smooth:
    :param word_given_class_smooth:
    :return: gold_label: a string of label provided with test data, winner: a predicted label (string),
    prob_class_given_doc for printing sys_out
    '''
    gold_label, word_num_dict = sample
    #for a given sample argmax P(c)*P(doc|class)
    winner, results = None, defaultdict(float) #to store results in for max
    # restrict records to ONLY ones in vocabulary
    word_keys = set(word_num_dict) & all_vocab
    for class_label in class_label_list:
        cond_array = []
        for key in word_keys:
            w, n = key, word_num_dict[key]
            prob, logprob = cond_probs[class_label].get(w, word_given_class_smooth[class_label])
            cond_array.append(n*logprob)
            #back to addition for logs since now using
        results[class_label] = class_priors[class_label][1] + np.array(cond_array).sum()
    winner = max(results, key=results.get)
    #calc prob class given sample, to return confidences for each class

    #fanciness below to avoid summing logs but also avoid underflow. 1) subtract the winner (the largest number) from
    #each log value. 2) take the absolute value of the smallest number and add to everything 3) raise 10^val so that can
    #sum the values. then divide as normal.
    step_1 = np.array(list(results.values())) - results[winner] #sum of each of the results
    min_val = abs(np.amin(step_1))
    #step_2 = step_1 + min_val
    #print(step_2)
    step_3 = [pow(10, val) for val in step_1]
    class_doc_joint_totals = sum(step_3)
    prob_class_given_doc = []
    for class_label in class_label_list:
        #I should really store this information when I calculate the sum to avoid recalculating it every time...
        numerator = pow(10, (results[class_label] - results[winner])) #+ min_val))
        prob_class_given_doc.append((class_label, numerator/class_doc_joint_totals))

    return gold_label, winner, prob_class_given_doc


if __name__ == "__main__":
    training_filename, testing_filename = sys.argv[1], sys.argv[2]
    class_prior_delta, cond_prob_delta = float(sys.argv[3]), float(sys.argv[4])
    model_filename, output_filename = sys.argv[5], sys.argv[6]
    #build model from training data
    class_priors, cond_probs, class_smooth_val, class_word_smooth_dict, train_records, vocab = \
                                            train_NB_Multinomial(training_filename, class_prior_delta, cond_prob_delta)
    #print model
    output_model(model_filename, class_priors, cond_probs)
    #testing stage
    class_label_list = sorted(list(class_priors))

    #testing phase
    gold_standard_train, predictions_train = [], []
    gold_standard_test, predictions_test = [], []
    sys_output_data = ['%%%%% training data:']
    #test on train
    for index in range(len(train_records)):
        record = train_records[index]
        gold, prediction, probs = test_NB_Multinomial(
            record, class_priors, cond_probs, class_label_list, vocab, class_smooth_val, class_word_smooth_dict)
        gold_standard_train.append(gold)
        predictions_train.append(prediction)
        prob_str = ' '.join(['{} {:.5f}'.format(t[0], t[1]) for t in sorted(probs, key=lambda x: x[1], reverse=True)])
        sys_output_data.append('array:{} {} {}'.format(index, gold, prob_str))
    #test on test
    array_num = 0
    sys_output_data.append('\n\n%%%%% test data:')
    with open(testing_filename, 'rU') as testfile:
        for line in testfile:
            sample = line.strip().split()
            doc_dict = Counter()
            for index in range(1, len(sample)):
                word, count = sample[index].split(':')
                doc_dict[word] =int(count)
                record = [sample[0], doc_dict]
            gold, prediction, probs = test_NB_Multinomial(
                    record, class_priors, cond_probs, class_label_list, vocab, class_smooth_val, class_word_smooth_dict)
            gold_standard_test.append(gold)
            predictions_test.append(prediction)
            prob_str = ' '.join(['{} {:.5f}'.format(t[0], t[1]) for t in sorted(probs, key=lambda x: x[1], reverse=True)])
            sys_output_data.append('array:{} {} {}'.format(array_num, gold, prob_str))
            array_num += 1
    #write to sys_output file
    with open(output_filename, 'w') as outfile:
        outfile.write('\n'.join(sys_output_data))
    #calc accuracy
    calc_accuracy(gold_standard_train, predictions_train, class_label_list, 'Training')
    calc_accuracy(gold_standard_test, predictions_test, class_label_list, 'Test')