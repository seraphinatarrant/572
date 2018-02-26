'''
A script that
Command to run: beamsearch_maxent.py test_data boundary_file model_file sys_output beam_size topN topK
'''

import argparse
from helpful_functions import read_maxent_model, maxent_classify_standard, calc_accuracy
from collections import deque, defaultdict
from math import log10
import numpy as np


class BSNode:

    def __init__(self, tag, tag_prob, path_prob, prev_node=None):
        self.path_prob = path_prob
        self.tag = tag
        self.tag_prob = tag_prob
        self.prev_node = prev_node

    def __str__(self):
        return '{} {}'.format(self.tag, self.path_prob)

    def get_tag_prob(self):
        return self.tag_prob

    def get_tag(self):
        return self.tag

    def get_prev_node(self):
        return self.prev_node

    def get_path_prob(self):
        return self.path_prob


def read_boundary_num(boundary_file):
    '''
    Processes a boundary file to get a deque of integers representing the length of each test set for beam search.
    :param boundary_file: expected to have one integer per line
    :return: a deque of ints
    '''
    b_num = deque()
    with open(boundary_file, 'r') as infile:
        for line in infile:
            num = int(line.strip())
            b_num.append(num)
    return b_num


def standard_to_binary_features(input_file, boundaries):
    '''
    :param input_file: A file in standard feature vector format
    :param boundaries: A deque of ints to delimit the boundaries of each sentence
    :return: a doubly nested list of processed training records of form:
    [ [[name, gold, feat_set],[name, gold, feat_set]], [[name, gold, feat_set],[name, gold, feat_set]] ]
    '''
    # can't regex the 1s out since '1' is a valid token that could appear in data. So easiest to split and skip them.
    all_records = []
    with open(input_file, 'r') as infile:
        curr_line, curr_max = 0, boundaries.popleft()
        curr_record = []
        for line in infile:
            if curr_line == curr_max:  # restart everything
                all_records.append(curr_record)
                curr_record, curr_line, curr_max = [], 0, boundaries.popleft()
            if line:
                line = line.strip().split()
                instanceName, gold_tag = line[0], line[1]
                feat_set = set()
                [feat_set.add(line[i]) for i in range(2, len(line), 2)] # create a feat set
                curr_record.append([instanceName, gold_tag, feat_set])
            curr_line += 1
        all_records.append(curr_record)  # this catches the last record
    return all_records


def backtrace(node, sequence):
    '''
    Traces back through beamsearch nodes by going to previous nodes and creates a string representation of the best path
    :param node: a BSNode
    :param sequence: The string representation of each node (a list)
    :return: the string sequence
    '''
    prev_node = node.get_prev_node()
    if prev_node.get_prev_node():
        sequence.append(prev_node)
        backtrace(prev_node, sequence)
    return reversed(sequence)


def beam_search_maxent(records, class_weights, feature_weights, all_classes, all_features, beam_size, topN, topK):
    '''

    :param test_records: a triply nested list, each index is a sentence, each index within that list is a word, which
    consists of arbitrary instanceName, gold label, and feature set
    :param c_weights: default class weights
    :param f_weights: default feature weights
    :param all_classes: list of all classes
    :param all_features: set of all features
    :param beam_size: beam size for pruning
    :param topN: topN to select each time expand nodes
    :param topK: topK winning paths to keep
    :return: A list of output data (in our standard sys out format), a list of gold labels, and a list of predictions
    '''

    output_data, gold_label_list, predictions = [], [], [] # for returning
    for record in records:  # each record is a single sentence of many words
        eos = len(record)
        node_state_table, prob_state_table = [[] for i in range(eos)], [[] for i in range(eos)]
        # initialise lists for tracking accuracy later
        labels = [record[0][0]]
        golds = [record[0][1]]  # instancename & gold label - later will zip together with results
        # initialise first word. Have to go one word at a time since the vector for the next word depends on previous
        classifications = maxent_classify_standard(record[0], class_weights, feature_weights, all_classes, all_features)  # returns pre-sorted and only topN
        # create BSNodes
        dummy_start_node = BSNode('BOS', 0.0, 0.0)  # since logprobs
        start_nodes = [BSNode(tag, tag_prob, tag_prob, dummy_start_node) for tag, tag_prob in classifications[:topN]]
        node_state_table[0] = start_nodes
        # iterate through the rest of the sentence word by word
        for i in range(1, eos):
            nodes_to_expand = node_state_table[i-1]  # this doesn't really have to be a copy since I won't need it to traverse...which means I could probably do it with a more efficient structure
            instancename, gold_label, feats = record[i]
            labels.append(instancename)
            golds.append(gold_label)
            all_next_nodes = []
            for j in range(len(nodes_to_expand)):
                prev_node = nodes_to_expand[j]
                prev_path_prob = prev_node.get_path_prob()
                prevTag, prev2Tag = prev_node.get_tag(), prev_node.get_prev_node().get_tag()
                # add tags to existing feature set
                new_feats = {'prevT={}'.format(prevTag), 'prevTwoTags={}+{}'.format(prev2Tag, prevTag)} | feats
                next_classes = maxent_classify_standard([instancename, gold_label, new_feats], class_weights,
                                                        feature_weights, all_classes, all_features)
                next_nodes = [BSNode(tag, tag_prob, prev_path_prob+tag_prob, prev_node) for tag, tag_prob in next_classes[:topN]] # I could pass topN to maxent classify and only return top N
                all_next_nodes.extend(next_nodes)
            # prune nodes
            prob_array = np.array([node.get_path_prob() for node in all_next_nodes])
            next_nodes_array = np.array(all_next_nodes)
            max_prob = np.max(prob_array) - beam_size
            if len(next_nodes_array) > topK:
                next_nodes_topK = next_nodes_array[np.argpartition(prob_array, -topK)[-topK:]]
            else:
                next_nodes_topK = next_nodes_array
            node_state_table[i] = [node for node in next_nodes_topK if node.get_path_prob() >= max_prob]
        # traverse winning path
        winner = max(node_state_table[eos-1], key=lambda x: x.get_path_prob())
        win_sequence = list(backtrace(winner, [winner]))
        output_data.extend(['{} {} {} {}'.format(labels[i], golds[i], win_sequence[i].get_tag(), 10**win_sequence[i].get_tag_prob()) for i in range(eos)])
        gold_label_list.extend(golds)
        predictions.extend([win_sequence[i].get_tag() for i in range(eos)])

    return output_data, gold_label_list, predictions



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('test_filename', help='A test file in standard feature vector format')
    p.add_argument('boundary_file', help='A file defining the boundaries of the sentence. One number per line, defining '
                                         'length of sentence in that line')
    p.add_argument('model_file', help='A MaxEnt .txt model file, of form "feature weight" one feature per line')
    p.add_argument('output_filename', help='the name for the sys out file')
    p.add_argument('beam_size', type=int, help='max gap between the lg-prob of the best path and the lg-prob of kept '
                                        'path: that is, a kept path should satisfy lg(prob)+beam_size >= lg(max_prob),'
                                        'where max_prob is the prob of the best path for the current position.')
    p.add_argument('topN', type=int, help='Num choices to keep when expanding a node on the beam search tree')
    p.add_argument('topK', type=int, help='Max num paths kept alive')
    args = p.parse_args()

    c_weights, f_weights, all_c, all_f = read_maxent_model(args.model_file)
    sent_lengths = read_boundary_num(args.boundary_file)
    test_records = standard_to_binary_features(args.test_filename, sent_lengths)
    sys_out, g, p = beam_search_maxent(test_records, c_weights, f_weights, all_c, all_f, args.beam_size, args.topN, args.topK) #COULD DO THIS ONE BY ONE RATHER THAN ALL AT ONCE...
    calc_accuracy(g, p, list(c_weights), 'test')

    with open(args.output_filename, 'w') as outfile:
        outfile.write('\n'.join(sys_out))


    