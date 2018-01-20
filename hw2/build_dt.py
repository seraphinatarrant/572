'''
A script that
Command to run: build_dt.sh training_data test_data max_depth min_gain model_file sys_output > acc_file

acc_file captures the print statements, which shows the confusion matrix and the accuracy for the training and test data

params:
training_data and testing_data: svm-light format feature vectors with one sample per line. in format "label word:count"
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
from sklearn.metrics import confusion_matrix


class DecisionTree:
    def __init__(self, root):
        self.root = root

    def get_root(self):
        return self.root

    def get_leaves(self):
        to_explore = deque([self.get_root()])
        leaf_nodes = set()
        while to_explore:
            current_node = to_explore.popleft()
            if not current_node.get_next():
                leaf_nodes.add(current_node)
            else:
                to_explore.extend(current_node.get_next())
        return leaf_nodes

    def print_model(self, target_file=sys.stdout):
        #gets paths for the decision tree leaves, starting at bottom
        leaves = self.get_leaves()
        root = self.get_root()
        output = []
        for leaf in leaves:
            path = deque()
            current = leaf
            while current != root:
                path.appendleft(str(current))
                current = current.get_prev()
            #also need to include number of samples, and probabilities for each class
            num_samples = len(leaf.get_samples())
            probs = ' '.join([str(value) for pair in leaf.sample_probabilities for value in pair])
            output.append('{} {} {}'.format('&'.join(path), num_samples, probs))
        with open(target_file, 'w') as outfile:
            outfile.write('\n'.join(sorted(output)))

    def set_leaf_probabilities(self):
        leaves = self.get_leaves()
        for leaf in leaves:
            #store probabilities of each class in a list of tuples of (class label, prob) based on samples at that node
            probs = []
            leaf_samples = leaf.get_samples()
            win_prob, win_class = 0, ''
            for label in classes:
                class_samples = set(training_class_dict[label]) & leaf_samples
                class_prob = len(class_samples)/len(leaf_samples)
                if class_prob > win_prob:
                    win_class = label
                    win_prob = class_prob
                class_prob_tuple = (label, class_prob)
                probs.append(class_prob_tuple)
            leaf.sample_probabilities = probs
            leaf.classification = win_class

    def classify_data(self, record):
        #used to classify new data once the tree is built
        #param record contains index 0 a gold standard class label, and index 1 a set of features
        gold_standard, features = record
        current_node = self.get_root()
        while current_node.get_next():
            next_label = current_node.get_action()
            if next_label in features:
                current_node = current_node.get_next()[0] #true feature
            else:
                current_node = current_node.get_next()[1] #false feature
        #now that have the leaf node, can return both the gold_standard, prediction, and probabilities
        return gold_standard, current_node.get_class(), current_node.get_string_probabilities()


class Node:
    def __init__(self, label, prev_node=None, depth=None, training_samples=set()):
        self.label = label
        self.next_nodes = [] #when this is initialised, it should always be true, false. This is a requirement.
        self.next_action = '' #used for running the decision tree once built
        self.prev_node = prev_node
        self.depth = depth
        self.training_samples = training_samples
        self.entropy = None
        self.path = ''
        self.sample_probabilities = [] #this will only ever be populated for leaf nodes, and will be a list of tuples
        self.classification = '' #this will only ever be populated for leaf nodes, and is argmax(sample probs)

    def __str__(self):
        return self.label

    def get_next(self):
        return self.next_nodes

    def get_action(self):
        return  self.next_action

    def get_prev(self):
        return self.prev_node

    def get_depth(self):
        return self.depth

    def get_samples(self):
        return self.training_samples

    def get_class(self):
        return self.classification

    def get_string_probabilities(self):
        if self.sample_probabilities:
            return ' '.join([str(value) for pair in self.sample_probabilities for value in pair])
        else:
            return ''

    def split_data(self, possible_features):
        winner, winning_info_gain = None, 0.0
        winning_true, winning_false = set(), set()
        # calculate entropy of current_node if not set
        if self.entropy == None:
            if len(self.training_samples) > 0:
                self.entropy = calc_entropy(self.training_samples) #should never be no data here...
            else:
                print('{} somehow has no training samples...'.format(self.label))
        # calculate infogain of splitting on each feature, and argmax
        for feature in possible_features:
            feature_true = set(training_word_dict[feature]) & self.training_samples
            feature_false = self.training_samples - feature_true
            #only bother to check entropy of the feature if it actually splits the data (so enforce min 1 sample in each)
            if feature_true and feature_false:
                #print(feature, feature_true, feature_false)
                new_info_gain = self.entropy - len(feature_true)/len(self.training_samples)*calc_entropy(feature_true) \
                            - len(feature_false)/len(self.training_samples)*calc_entropy(feature_false)
                if new_info_gain > winning_info_gain:
                    winner = feature
                    winning_info_gain = new_info_gain
                    winning_true = feature_true
                    winning_false = feature_false
                    #print('!!!!!!!!!!', winner, winning_info_gain, len(winning_true), len(winning_false))
        #print(self.entropy)
        #print('%%%%%%%%%%%', winner, winning_info_gain, len(winning_true), len(winning_false))
        return winner, winning_info_gain, winning_true, winning_false

def calc_entropy(data):
    if len(data) < 1:
        print("WTF no data I can't calculate entropy of nothing!")
        return 0
    distribution = []
    base = 2.0
    for label in classes:
        class_samples = set(training_class_dict[label]) & data
        class_prob = len(class_samples)/len(data)
        distribution.append(class_prob)
    return entropy(distribution, None, base)

def build_binary_decision_tree(training_word_dict, training_ids, max_depth, min_gain):
    '''
    Algorithm that builds a binary decision tree based on the given training data
    :param training_word_dict: a dictionary of {word: list of training ids that contain that word}
    :param training_ids: a set of total training ids
    :return:
    '''
    start = Node('start', None, 0, set(training_ids))
    binary_tree = DecisionTree(start)
    possible_features = list(training_word_dict) #possible features are all the words (keys) in training dict
    leaf_nodes = deque() #FIFO list of leaf nodes to explore
    leaf_nodes.append(start)
    while leaf_nodes:
        current_node = leaf_nodes.popleft()
        #restrict for maximum depth
        if current_node.get_depth() == max_depth:
            continue
        best_feature, best_gain, true_samples, false_samples = current_node.split_data(possible_features)
        #restrict for minimum gain
        if best_gain < min_gain:
            continue
        if not true_samples or not false_samples: #####ok so how does it ever return a winner with no samples? Winner would be none...
            ###SO I shouldn't have to do this check.
            continue
        #remove winning feature from possible_features
        possible_features.remove(best_feature)
        #initialise new nodes with the winning data split
        new_node_true = Node(best_feature, current_node, current_node.depth+1, true_samples)
        new_node_false = Node('!{}'.format(best_feature), current_node, current_node.depth+1, false_samples)
        #update current node to point to this as next level
        current_node.next_nodes = [new_node_true, new_node_false]
        current_node.next_action = best_feature #this just stores the positive version for use when running tree
        #add new nodes to leaf nodes
        if len(true_samples) > 1:
            leaf_nodes.append(new_node_true)
        if len(false_samples) > 1:
            leaf_nodes.append(new_node_false)
    binary_tree.set_leaf_probabilities()
    return binary_tree

def calc_accuracy(gold_standard, predictions, type):
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
    print('\n{} accuracy={}\n\n'.format(type, accuracy))


if __name__ == "__main__":
    training_data_filename, test_data_filename = sys.argv[1], sys.argv[2]
    max_depth, min_gain = int(sys.argv[3]), float(sys.argv[4])
    model_filename, sys_output_filename = sys.argv[5], sys.argv[6]

    ####read in files and initialise everything
    #make training_word_dict
    global training_word_dict
    training_word_dict = defaultdict(list)
    #make training_class_dict of {class: training ids}
    global training_class_dict
    training_class_dict = defaultdict(list)
    training_records_list = [] #this is for testing on the training data
    #read in training data in structures to support training & also build the list of sets to test on once tree is built
    with open(training_data_filename, 'rU') as trainfile:
        doc_id = 0
        for line in trainfile:
            sample = line.strip().split() # idx 0 is the label, all others are word:count
            training_class_dict[sample[0]].append(doc_id)
            doc_set = set()
            for index in range(1, len(sample)):
                word, count = sample[index].split(':')
                training_word_dict[word].append(doc_id)
                doc_set.add(word)
            doc_id += 1
            training_records_list.append([sample[0], doc_set]) #this is a list [label, set of words]
    #initalise a global of all the classes read in
    global classes
    classes = sorted(list(training_class_dict))
    #build the binary tree and print the model
    binary_DT = build_binary_decision_tree(training_word_dict, [num for num in range(doc_id)], max_depth, min_gain)
    binary_DT.print_model(model_filename)

    # TODO yes the training and testing output is a little copy-pasty, could make them a function with flags
    #two lists to keep track of gold_standard and predicted classifications, for use in calculating accuracy at end
    gold_standard_train, predictions_train = [], []
    sys_output_data = ['%%%%% training data:']
    #test on the training data
    for index in range(len(training_records_list)):
        record = training_records_list[index]
        gold, prediction, probs = binary_DT.classify_data(record)
        gold_standard_train.append(gold)
        predictions_train.append(prediction)
        sys_output_data.append('array:{} {}'.format(index, probs))
    # test on the test data - can do this line by line as it comes in (faster, if there are a lot of documents
    sys_output_data.append('\n\n%%%%% test data:')
    gold_standard_test, predictions_test = [], []
    array_num = 0
    with open(test_data_filename, 'rU') as testfile:
        for line in testfile:
            sample = line.strip().split()
            doc_set = set()
            for index in range(1, len(sample)):
                word, count = sample[index].split(':')
                doc_set.add(word)
            gold, prediction, probs = binary_DT.classify_data([sample[0], doc_set])
            gold_standard_test.append(gold)
            predictions_test.append(prediction)
            sys_output_data.append('array:{} {}'.format(array_num, probs))
            array_num += 1
    #write to sys_output file
    with open(sys_output_filename, 'w') as outfile:
        outfile.write('\n'.join(sys_output_data))
    #calc accuracy
    calc_accuracy(gold_standard_train, predictions_train, 'Training')
    calc_accuracy(gold_standard_test, predictions_test, 'Test')

