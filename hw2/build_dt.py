'''
A script that
Command to run: build_dt.sh training_data test_data max_depth min_gain model_file sys_output > acc_file

acc_file captures the print statements, which shows the confusion matrix and the accuracy for the training and test data

params:
max_depth: is max depth of DT (root is zero)...so max depth is number of times split I believe.
min_gain: is min info gain.
model_file: DT tree produced by the trainer, in format: path training_instance_num c1 p1 c2 p2
    e.g. !gun&!com&!how&israel 5 talk.politics.guns 0 talk.politics.misc 0 talk.politics.mideast 1
    where path is the path to get to the leaf node, then num of instances there, then labels and probabilities
sys_output: is the classification result on the training and test data with instanceName (array:0, arrray:1, etc) and classes with probs
    e.g. array:0	talk.politics.guns	0.9823325116912758	talk.politics.mideast	0.0068180812775302255	talk.politics.misc	0.010849407031193997

Features are binary presence of words
Split on each feature
Function to evaluate features is info gain

'''

import sys
from collections import defaultdict, deque
from scipy.stats import entropy

class DecisionTree:
    def __init__(self, root):
        self.root = root

    def get_root(self):
        return self.root

class Node:
    def __init__(self, label, prev_node=None, depth=None, training_samples=set()):
        self.label = label
        self.next_nodes = []
        self.prev_node = prev_node
        self.depth = depth
        self.training_samples = training_samples
        self.entropy = None

    def get_next(self):
        return self.next_nodes

    def get_prev(self):
        return self.prev_node

    def get_depth(self):
        return self.depth

    def split_data(self, possible_features):
        winner, winning_info_gain = None, 0.0
        # calculate entropy of current_node if not set
        if self.entropy == None:
            self.entropy = calc_entropy(self.training_samples)
        # calculate infogain of splitting on each feature, and argmax
        for feature in possible_features:
            feature_true = set(training_word_dict[feature]) & self.training_samples
            feature_false = self.training_samples - feature_true
            new_info_gain = self.entropy - len(feature_true) / len(self.training_samples)*calc_entropy(feature_true) - \
                            len(feature_false) / len(self.training_samples) * calc_entropy(feature_false)
            if new_info_gain > winning_info_gain:
                winner = feature
                winning_info_gain = new_info_gain
        return winner, winning_info_gain, feature_true, feature_false

def calc_entropy(data):
    distribution = []
    base = 2.0
    for label in classes:
        class_samples = set(training_class_dict[label]) & data
        class_prob = len(class_samples)/len(data)
        distribution.append(class_prob)
    return entropy(distribution, None, base)

def build_binary_decision_tree(training_word_dict, training_ids, max_depth, min_gain):
    '''
    Algorithm that builds a binary decision tree based on
    :param training_word_dict: a dictionary of {word: list of training ids that contain that word}
    :param training_ids: a set of total training ids
    :return:
    '''
    # DT_dict is dict of nested lists. Idx 0 is parent node name, Idx 1 is depth, idx 2 is training IDs at this node
    start = Node('start', None, 0, training_ids)
    binary_tree = DecisionTree(start)
    possible_features = list(training_word_dict) #possible features are all the words (keys) in training dict
    leaf_nodes = deque(start) #FIFO list of leaf nodes to explore
    while leaf_nodes:
        current_node = leaf_nodes.popleft()
        #restrict for maximum depth
        if current_node.get_depth() == max_depth:
            continue
        best_feature, best_gain, true_samples, false_samples = current_node.split_data(possible_features)
        #restrict for minimum gain
        if best_gain < min_gain:
            continue
        #remove winning feature from possible_features
        possible_features.remove(best_feature)
        #initialise new nodes with the winning data split
        new_node_true = Node(best_feature, current_node, current_node.depth+1, true_samples)
        new_node_false = Node('!{}'.format(best_feature), current_node, current_node.depth+1, false_samples)
        #update current node to point to this as next level
        current_node.next_nodes = [new_node_true, new_node_false]
        #add new nodes to leaf nodes
        leaf_nodes.extend([new_node_true, new_node_false])

'''
Once Decision Tree is Built:
Print the model file
Run test data
Calc accuracy
'''



if __name__ == "__main__":
    training_data_filename, test_data_filename = sys.argv[1], sys.argv[2]
    max_depth, min_gain = int(sys.argv[3]), float(sys.argv[4])
    model_filename, sys_output_filename = sys.argv[5], sys.argv[6]

    #make training_word_dict
    training_word_dict = defaultdict(list)
    #make training_class_dict of {class: training ids}
    training_class_dict = defaultdict(list)
    #make dict for DT
    DT_dict = defaultdict(list)

    global classes
    classes = list(training_class_dict)


