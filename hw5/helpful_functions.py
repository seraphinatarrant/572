from sklearn.metrics import confusion_matrix
import numpy as np

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



if __name__ == "__main__":
    print('Hello I am only a functions helper file! Love me anyway!')
    